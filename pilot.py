"""Prepare and analyze a lot-grouped feasibility pilot for honey screening."""

import argparse
import json
from pathlib import Path

from benchmark import SEED, decision_score, fit_model, prepare_inner, snv, transform, tune

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from datasets import GRID_NM, resample
from followup import (
    LABEL_FIELDS, MODELS, SCAN_FIELDS, aggregate_scans, exact_links, exact_lots, features,
    fraction, input_frame, json_bytes, load_labels, read_scan_file, sha256, write_new,
)
from followup_stats import ENDPOINTS, complete_panel, lot_success, threshold_for_specificity
from models import SEARCH


SCHEDULE_FIELDS = ["day", "order", *SCAN_FIELDS]


def write_csv(path, rows):
    write_new(path, rows.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def make_plan(lots, adulterants, instrument, repeat_count, seed):
    if len(set(lots)) != len(lots) or len(lots) < 5 or not 0 <= repeat_count <= len(lots):
        raise ValueError("Provide at least five distinct lots and a valid repeat-lot count.")
    if len(set(adulterants)) != len(adulterants) or "none" in adulterants:
        raise ValueError("Use distinct adulterant names and reserve 'none' for pure honey.")
    rng = np.random.default_rng(seed)
    panel = [(0, "none")] + [(dose, syrup) for dose in (5, 10) for syrup in adulterants]
    labels = []
    for lot in sorted(lots):
        for dose, syrup in panel:
            labels.append({"lot_id": lot, "dose_pct": dose, "adulterant": syrup})
    labels = pd.DataFrame(labels)
    labels["preparation_id"] = [f"P{value:012x}" for value in rng.choice(2**48, len(labels), replace=False)]
    repeat_lots = rng.choice(sorted(lots), size=repeat_count, replace=False)
    repeat = labels.loc[labels["lot_id"].isin(repeat_lots)]
    schedules = []
    for day, rows in (("day_1", labels), ("day_2", repeat)):
        shuffled = rows.iloc[rng.permutation(len(rows))][["lot_id", "preparation_id"]].copy()
        shuffled["order"] = np.arange(1, len(rows) + 1)
        shuffled["day"] = day
        shuffled["instrument"] = instrument
        shuffled["scan_id"] = [f"{day}-{i:04d}" for i in shuffled["order"]]
        schedules.append(shuffled[SCHEDULE_FIELDS])
    return labels[LABEL_FIELDS], pd.concat(schedules, ignore_index=True)


def prepare(args):
    if not args.instrument.strip() or args.instrument != args.instrument.strip():
        raise ValueError("Provide a nonempty instrument ID without surrounding whitespace.")
    if any(not name.strip() or name != name.strip() for name in args.adulterants):
        raise ValueError("Adulterant names must be nonempty and without surrounding whitespace.")
    lots = (input_frame(args.lot_ids, ["lot_id"])["lot_id"].tolist() if args.lot_ids
            else [f"PILOT-{i:02d}" for i in range(1, args.lots + 1)])
    labels, schedule = make_plan(lots, args.adulterants, args.instrument, args.repeat_lots, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "lot_inventory.csv", pd.DataFrame({"lot_id": sorted(lots)}))
    write_csv(args.out / "preparation_key.csv", labels)
    write_csv(args.out / "scan_schedule.csv", schedule)
    design = {
        "instrument": args.instrument, "seed": args.seed, "adulterants": sorted(args.adulterants),
        "lot_count": len(lots), "preparation_count": len(labels), "repeat_lot_count": args.repeat_lots,
        "planned_repeat_scans": int(schedule["day"].eq("day_2").sum()),
        "lot_ids": "supplied" if args.lot_ids else "placeholders",
    }
    write_new(args.out / "design.json", json_bytes(design))
    print(json_bytes(design).decode().rstrip())


def inspect_scan(meta, values, wavelengths):
    if not np.isfinite(values).all() or not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Expected finite signals and a strictly increasing wavelength axis.")
    if meta["scan_id"].duplicated().any():
        raise ValueError("Scan IDs must be unique within an export.")
    if np.any(values.std(axis=1) < 1e-12):
        raise ValueError("At least one exported spectrum is constant.")
    return {
        "instrument": meta["instrument"].iat[0], "scans": len(meta), "native_channels": len(wavelengths),
        "min_nm": float(wavelengths[0]), "max_nm": float(wavelengths[-1]),
        "covers_780_1000_nm": bool(wavelengths[0] <= GRID_NM[0] and wavelengths[-1] >= GRID_NM[-1]),
    }


def check_export(args):
    print(json_bytes(inspect_scan(*read_scan_file(args.spectra))).decode().rstrip())


def load_collection(plan, baseline, repeat):
    design = json.loads((plan / "design.json").read_bytes())
    schedule = input_frame(plan / "scan_schedule.csv", SCHEDULE_FIELDS)
    labels = load_labels(plan / "preparation_key.csv")
    inventory = input_frame(plan / "lot_inventory.csv", ["lot_id"])
    if inventory["lot_id"].duplicated().any() or len(inventory) != design["lot_count"] or len(labels) != design["preparation_count"]:
        raise ValueError("The inventory or preparation count differs from the pilot design.")
    exact_lots(labels, inventory["lot_id"], "pilot inventory")
    complete_panel(labels, design["adulterants"])
    if not set(schedule["day"]).issubset({"day_1", "day_2"}) or schedule["scan_id"].duplicated().any():
        raise ValueError("The schedule must contain distinct scan IDs on day_1 or day_2.")
    day_1 = schedule.loc[schedule["day"].eq("day_1")]
    exact_links(day_1, labels, "day-one preparation key")
    day_2 = schedule.loc[schedule["day"].eq("day_2")]
    exact_links(day_2, labels.loc[labels["lot_id"].isin(day_2["lot_id"])], "day-two preparation key")
    if len(day_2) != design["planned_repeat_scans"] or day_2["lot_id"].nunique() != design["repeat_lot_count"]:
        raise ValueError("The second-day schedule differs from the pilot design counts.")
    exports = {}
    for day, path in (("day_1", baseline), ("day_2", repeat)):
        if path is None:
            continue
        export = read_scan_file(path)
        info = inspect_scan(*export)
        expected = schedule.loc[schedule["day"].eq(day), SCAN_FIELDS]
        if info["instrument"] != design["instrument"] or set(export[0].itertuples(index=False, name=None)) != set(expected.itertuples(index=False, name=None)):
            raise ValueError(f"{day} export scan, lot, preparation or instrument IDs differ from the schedule.")
        exports[day] = export
    if repeat and not np.array_equal(exports["day_1"][2], exports["day_2"][2]):
        raise ValueError("Day-one and day-two exports must use the same instrument wavelength axis.")
    return labels, exports, design


def build_folds(groups, outer_folds, calibration_groups, seed, inner_folds=3):
    names = np.unique(groups)
    largest_test = (len(names) + outer_folds - 1) // outer_folds
    if not 2 <= outer_folds <= len(names) or not 1 <= calibration_groups <= len(names) - largest_test - inner_folds:
        raise ValueError(f"Each fold needs separate test and calibration groups plus at least {inner_folds} development groups.")
    rng = np.random.default_rng(seed)
    for test in np.array_split(rng.permutation(names), outer_folds):
        available = np.setdiff1d(names, test)
        calibration = rng.choice(available, calibration_groups, replace=False)
        development = np.setdiff1d(available, calibration)
        yield [np.isin(groups, names) for names in (development, calibration, test)]


def group_ids(labels, path):
    if path is None:
        return labels["lot_id"].to_numpy()
    mapping = input_frame(path, ["lot_id", "validation_group"])
    exact_lots(mapping, labels["lot_id"], "validation grouping")
    if mapping["lot_id"].duplicated().any():
        raise ValueError("Each lot must have one validation group.")
    return labels["lot_id"].map(mapping.set_index("lot_id")["validation_group"]).to_numpy()


def spectra_for_window(export, grid, labels):
    meta, values, wavelengths = export
    scans = pd.concat([meta, pd.DataFrame(resample(values, wavelengths, grid))], axis=1)
    return aggregate_scans(scans).merge(labels, on=["lot_id", "preparation_id"], validate="one_to_one")


def spectral_repeatability(baseline, repeat, grid):
    if repeat.empty:
        return None
    original = baseline.set_index("preparation_id").loc[repeat["preparation_id"]]
    delta = snv(features(original, grid)) - snv(features(repeat, grid))
    rms = np.sqrt(np.mean(delta**2, axis=1))
    return {"preparations": len(rms), "median_snv_rms": float(np.median(rms)), "p95_snv_rms": float(np.quantile(rms, 0.95))}


def predictions_for_window(window, grid, labels, exports, args, seed):
    baseline = spectra_for_window(exports["day_1"], grid, labels)
    repeat = spectra_for_window(exports["day_2"], grid, labels) if "day_2" in exports else baseline.iloc[:0]
    groups = group_ids(baseline, args.validation_groups)
    x, y = features(baseline, grid), baseline["label"].to_numpy()
    records, fold_log = [], []
    splits = build_folds(groups, args.outer_folds, args.calibration_groups, seed, args.inner_folds)
    for fold, (dev, calibration, test) in enumerate(splits, start=1):
        fold_seed = seed + 100 * fold
        inner = prepare_inner(x[dev], y[dev], groups[dev], args.inner_folds)
        scaler = StandardScaler().fit(snv(x[dev]))
        train, pure = transform(scaler, x[dev]), transform(scaler, x[calibration & (y == 0)])
        held_out = baseline.loc[test]
        repeat_test = repeat.loc[repeat["preparation_id"].isin(held_out["preparation_id"])]
        phases = [("day_1", held_out, transform(scaler, x[test]))]
        if not repeat_test.empty:
            phases.append(("day_2", repeat_test, transform(scaler, features(repeat_test, grid))))
        model_log = {}
        for name in MODELS:
            index, candidates, _ = tune(name, y[dev], inner, fold_seed)
            params = SEARCH[name][index]
            model = fit_model(name, params, train, y[dev], fold_seed + 50)
            controls = decision_score(name, model, pure)
            threshold = threshold_for_specificity(controls, args.specificity)
            model_log[name] = {"params": params, "inner_scores": candidates, "threshold": threshold,
                               "calibration_pure_lots": int(len(controls)),
                               "calibration_specificity": float(np.mean(controls <= threshold))}
            for day, meta, values in phases:
                scores = decision_score(name, model, values)
                if not np.isfinite(scores).all():
                    raise ValueError(f"Nonfinite scores for {name} in fold {fold}.")
                records.append(meta[LABEL_FIELDS].assign(
                    window=window, fold=fold, day=day, model=name, score=scores,
                    threshold=threshold, prediction=(scores > threshold).astype(int),
                ))
        fold_log.append({"window": window, "fold": fold, "seed": fold_seed, "models": model_log,
                         "development_groups": sorted(set(groups[dev])), "calibration_groups": sorted(set(groups[calibration])),
                         "test_groups": sorted(set(groups[test])),
                         "inner_validation_groups": [sorted(set(groups[dev][part[3]])) for part in inner]})
    return pd.concat(records, ignore_index=True), fold_log, spectral_repeatability(baseline, repeat, grid)


def paired_repeats(baseline, repeat):
    joined = baseline.merge(repeat, on=["lot_id", "preparation_id", "fold"], suffixes=("_1", "_2"), validate="one_to_one")
    if len(joined) != len(repeat):
        raise ValueError("Every second-day score must pair with a first-day score from the same fold.")
    if joined.empty:
        return None
    flips = joined["prediction_1"].ne(joined["prediction_2"])
    return {"preparations": len(joined), "lots": joined["lot_id"].nunique(), "decision_flips": int(flips.sum()),
            "decision_agreement": float(1 - flips.mean()),
            "median_absolute_score_change": float((joined["score_2"] - joined["score_1"]).abs().median())}


def results_summary(predictions, targets):
    summaries = []
    for (window, model), rows in predictions.groupby(["window", "model"], sort=False):
        baseline = rows.loc[rows["day"].eq("day_1")].sort_values("preparation_id")
        success = lot_success(baseline)
        values = {name: {"successes": int(success[name].sum()), "lots": len(success),
                         "estimate": float(success[name].mean()), "target": targets[name]}
                  for name in ENDPOINTS}
        syrup = baseline.loc[baseline["dose_pct"].ne(0)].groupby(["adulterant", "dose_pct"])["prediction"].agg(["sum", "size"])
        summaries.append({
            "window": window, "model": model, "endpoints": values,
            "meets_all_pilot_point_targets": all(value["estimate"] >= value["target"] for value in values.values()),
            "per_syrup": [{"adulterant": a, "dose_pct": float(d), "detected": int(row["sum"]), "preparations": int(row["size"])}
                          for (a, d), row in syrup.iterrows()],
            "second_day": paired_repeats(baseline, rows.loc[rows["day"].eq("day_2")]),
        })
    return summaries


def analyze(args):
    if args.out.exists():
        raise ValueError("Use a new output directory for each pilot analysis.")
    labels, exports, design = load_collection(args.plan, args.baseline_spectra, args.repeat_spectra)
    info = inspect_scan(*exports["day_1"])
    native = np.linspace(info["min_nm"], info["max_nm"], len(GRID_NM))
    windows = {"native_span": native}
    if info["covers_780_1000_nm"] and not np.array_equal(native, GRID_NM):
        windows = {"common_780_1000_nm": GRID_NM, **windows}
    records, folds, repeatability = [], [], {}
    for window, grid in windows.items():
        prediction, log, repeat_stats = predictions_for_window(window, grid, labels, exports, args, design["seed"])
        records.append(prediction)
        folds.extend(log)
        repeatability[window] = repeat_stats
    predictions = pd.concat(records, ignore_index=True)
    targets = dict(zip(ENDPOINTS, (args.specificity, args.sensitivity_5, args.sensitivity_10)))
    paths = {"baseline_spectra": args.baseline_spectra, "repeat_spectra": args.repeat_spectra,
             "validation_groups": args.validation_groups}
    paths.update({name: args.plan / name for name in ("design.json", "scan_schedule.csv", "preparation_key.csv", "lot_inventory.csv")})
    summary = {
        "design": design, "instrument_export": info, "grids_nm": {key: value.tolist() for key, value in windows.items()},
        "models": results_summary(predictions, targets),
        "spectral_repeatability": repeatability,
        "repeat_status": "analyzed" if args.repeat_spectra else ("pending" if design["repeat_lot_count"] else "not_scheduled"),
        "sha256": {name: sha256(path.read_bytes()) for name, path in paths.items() if path},
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "predictions.csv", predictions)
    write_new(args.out / "folds.json", json_bytes(folds))
    write_new(args.out / "summary.json", json_bytes(summary))
    for result in summary["models"]:
        rates = " | ".join(f"{name} {value['successes']}/{value['lots']}" for name, value in result["endpoints"].items())
        print(f"{result['window']} | {result['model']} | {rates}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("prepare", help="Generate the preparation key and randomized scan schedule.")
    planning.add_argument("--instrument", required=True)
    planning.add_argument("--adulterants", nargs="+", default=["rice", "sucrose"])
    planning.add_argument("--lots", type=int, default=24)
    planning.add_argument("--lot-ids", type=Path)
    planning.add_argument("--repeat-lots", type=int, default=6)
    planning.add_argument("--seed", type=int, default=SEED)
    planning.add_argument("--out", type=Path, required=True)
    checking = commands.add_parser("check-export", help="Check one or more rows from an instrument export.")
    checking.add_argument("--spectra", type=Path, required=True)
    evaluation = commands.add_parser("analyze", help="Compare spectral windows and score repeat-day stability.")
    evaluation.add_argument("--plan", type=Path, required=True)
    evaluation.add_argument("--baseline-spectra", type=Path, required=True)
    evaluation.add_argument("--repeat-spectra", type=Path)
    evaluation.add_argument("--validation-groups", type=Path)
    evaluation.add_argument("--outer-folds", type=int, default=6, choices=range(2, 25), metavar="2..24")
    evaluation.add_argument("--inner-folds", type=int, default=3, choices=range(2, 6))
    evaluation.add_argument("--calibration-groups", type=int, default=10)
    evaluation.add_argument("--specificity", type=fraction, default=0.90)
    evaluation.add_argument("--sensitivity-5", type=fraction, default=0.80)
    evaluation.add_argument("--sensitivity-10", type=fraction, default=0.80)
    evaluation.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "analyze":
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            with threadpool_limits(limits=1):
                analyze(args)
        else:
            {"prepare": prepare, "check-export": check_export}[args.command](args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
