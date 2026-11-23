"""Plan, freeze, score and evaluate a blinded lot-level validation study."""

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

from benchmark import SEED, decision_score, fit_model, prepare_inner, snv, transform, tune

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from datasets import GRID_NM, resample
from followup_stats import (
    ENDPOINTS, complete_panel, lot_success, rate_summary, sample_size_plan, threshold_for_specificity,
)
from models import SEARCH


MODELS = ("SVM", "XGBoost", "MLP")
PARTITIONS = {"development", "calibration", "test"}
SCAN_FIELDS = ["lot_id", "preparation_id", "instrument", "scan_id"]
LABEL_FIELDS = ["lot_id", "preparation_id", "dose_pct", "adulterant"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def input_frame(path, required, spectral=False):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    extra = set(frame) - set(required)
    if not set(required).issubset(frame) or (extra and not spectral):
        raise ValueError(f"{path} must have fields {required}.")
    if frame.empty or any(not frame[col].str.strip().eq(frame[col]).all() or frame[col].eq("").any() for col in required):
        raise ValueError(f"{path} contains empty fields or surrounding whitespace.")
    return frame


def load_partitions(path):
    frame = input_frame(path, ["lot_id", "partition"])
    if frame["lot_id"].duplicated().any() or set(frame["partition"]) != PARTITIONS:
        raise ValueError("Each lot must appear once, with development, calibration and test lots represented.")
    return {name: sorted(frame.loc[frame["partition"].eq(name), "lot_id"]) for name in PARTITIONS}


def load_labels(path):
    frame = input_frame(path, LABEL_FIELDS)
    frame["dose_pct"] = pd.to_numeric(frame["dose_pct"], errors="raise")
    if frame["preparation_id"].duplicated().any() or not frame["dose_pct"].between(0, 100, inclusive="left").all():
        raise ValueError("Labels require a unique preparation ID and a dose in [0, 100).")
    if not frame["adulterant"].eq("none").equals(frame["dose_pct"].eq(0)):
        raise ValueError("Use adulterant 'none' exactly for pure preparations at dose 0.")
    frame["label"] = frame["dose_pct"].gt(0).astype(int)
    return frame


def read_scan_file(path):
    frame = input_frame(path, SCAN_FIELDS, spectral=True)
    bands = [name for name in frame if name not in SCAN_FIELDS]
    if len(bands) < 2 or any(not re.fullmatch(r"\d+(?:\.\d+)?nm", band) for band in bands):
        raise ValueError(f"{path} must contain only scan identifiers and wavelength columns named like 980.0nm.")
    if frame["instrument"].nunique() != 1:
        raise ValueError(f"Use one instrument per spectral CSV: {path}.")
    bands.sort(key=lambda band: float(band[:-2]))
    wavelengths = np.array([float(band[:-2]) for band in bands])
    return frame[SCAN_FIELDS], frame[bands].to_numpy(dtype=float), wavelengths


def aggregate_scans(scans):
    if scans.duplicated(["instrument", "scan_id"]).any():
        raise ValueError("Scan IDs must be unique within each instrument.")
    if scans.groupby("preparation_id")["lot_id"].nunique().gt(1).any():
        raise ValueError("Every preparation must belong to exactly one lot across instruments.")
    return scans.drop(columns="scan_id").groupby(SCAN_FIELDS[:3], sort=True).mean().reset_index()


def load_spectra(paths, grid):
    scans = []
    for path in paths:
        meta, values, wavelengths = read_scan_file(path)
        x = resample(values, wavelengths, grid)
        scans.append(pd.concat([meta, pd.DataFrame(x)], axis=1))
    return aggregate_scans(pd.concat(scans, ignore_index=True))


def exact_links(left, right, context):
    key = ["lot_id", "preparation_id"]
    if set(left[key].itertuples(index=False, name=None)) != set(right[key].itertuples(index=False, name=None)):
        raise ValueError(f"Preparation and lot IDs do not match for {context}.")


def features(frame, grid):
    return frame[list(range(len(grid)))].to_numpy(dtype=float)


def exact_lots(frame, expected, context):
    missing, extra = set(expected) - set(frame["lot_id"]), set(frame["lot_id"]) - set(expected)
    if missing or extra:
        raise ValueError(f"Unexpected lot coverage for {context}. Missing: {sorted(missing)}. Unexpected: {sorted(extra)}.")


def write_new(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(contents)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def freeze(args):
    if args.out.exists() or args.out.with_suffix(".json").exists() or args.out.suffix != ".pkl":
        raise ValueError("Choose a new .pkl path for the frozen model and its JSON receipt.")
    if sorted(set(args.adulterants)) != sorted(args.adulterants) or "none" in args.adulterants:
        raise ValueError("Supply distinct test adulterants and reserve 'none' for pure controls.")
    if not np.isfinite([args.min_nm, args.max_nm]).all() or not 0 < args.min_nm < args.max_nm:
        raise ValueError("Supply a positive, ascending wavelength range.")
    grid = np.linspace(args.min_nm, args.max_nm, len(GRID_NM))
    target = args.target_instrument or args.source_instrument
    calibration_instrument = args.calibration_instrument or target
    if calibration_instrument not in {args.source_instrument, target}:
        raise ValueError("Calibration must use the source or target instrument.")
    partitions = load_partitions(args.partitions)
    if args.pilot_lots:
        pilot_lots = input_frame(args.pilot_lots, ["lot_id"])
        overlap = set(partitions["test"]) & set(pilot_lots["lot_id"])
        if overlap:
            raise ValueError(f"Pilot lots cannot appear in the final test partition: {sorted(overlap)}.")
    allowed_lots = sorted(partitions["development"] + partitions["calibration"])
    labels = load_labels(args.labels)
    exact_lots(labels, allowed_lots, "freeze labels")
    scans = load_spectra(args.spectra, grid)
    exact_lots(scans, allowed_lots, "freeze spectra")
    exact_links(scans, labels, "development and calibration spectra")
    scans = scans.merge(labels, on=["lot_id", "preparation_id"], validate="many_to_one")
    development = scans.loc[scans["lot_id"].isin(partitions["development"]) & scans["instrument"].eq(args.source_instrument)]
    calibration = scans.loc[scans["lot_id"].isin(partitions["calibration"]) & scans["instrument"].eq(calibration_instrument) & scans["label"].eq(0)]
    exact_lots(development, partitions["development"], "source development spectra")
    exact_lots(calibration, partitions["calibration"], "calibration pure controls")
    exact_links(development, labels.loc[labels["lot_id"].isin(partitions["development"])], "source development spectra")
    if calibration["lot_id"].duplicated().any():
        raise ValueError("Use one pure calibration preparation per lot. Average repeated scans via the same preparation ID.")

    y = development["label"].to_numpy()
    groups = development["lot_id"].to_numpy()
    x = features(development, grid)
    inner = prepare_inner(x, y, groups)
    scaler = StandardScaler().fit(snv(x))
    train = transform(scaler, x)
    controls = transform(scaler, features(calibration, grid))
    learners, model_details = {}, {}
    for name in MODELS:
        candidate, scores, _ = tune(name, y, inner, SEED)
        params = SEARCH[name][candidate]
        learner = fit_model(name, params, train, y, SEED + 50)
        margins = decision_score(name, learner, controls)
        threshold = threshold_for_specificity(margins, args.specificity)
        learners[name] = learner
        model_details[name] = {
            "params": params, "development_candidate_balanced_accuracies": scores,
            "threshold": threshold, "calibration_pure_lots": len(margins),
            "observed_calibration_specificity": float(np.mean(margins <= threshold)),
        }
    receipt = {
        "schema_version": 1, "primary_model": args.primary_model, "alpha": args.alpha,
        "specificity": args.specificity, "sensitivity_5pct": args.sensitivity_5,
        "sensitivity_10pct": args.sensitivity_10, "adulterants": sorted(args.adulterants),
        "source_instrument": args.source_instrument, "target_instrument": target,
        "calibration_instrument": calibration_instrument, "partitions": partitions,
        "training_preparations": sorted(labels["preparation_id"]),
        "wavelengths_nm": grid.tolist(), "seed": SEED, "models": model_details,
        "development_inner_validation_lots": [sorted(set(groups[part[3]])) for part in inner],
        "inputs": {str(path): sha256(path.read_bytes()) for path in [args.partitions, args.labels, *args.spectra, args.pilot_lots] if path},
    }
    bundle = pickle.dumps({"receipt": receipt, "scaler": scaler, "learners": learners}, protocol=5)
    write_new(args.out, bundle)
    write_new(args.out.with_suffix(".json"), json_bytes({**receipt, "frozen_sha256": sha256(bundle)}))
    print(f"Frozen {args.primary_model} as the primary model. SHA-256 {sha256(bundle)}")


def score(args):
    data = args.frozen.read_bytes()
    bundle = pickle.loads(data)
    receipt = bundle["receipt"]
    grid = np.array(receipt["wavelengths_nm"])
    x = load_spectra(args.spectra, grid)
    instruments = sorted({receipt["source_instrument"], receipt["target_instrument"]})
    if set(x["instrument"]) != set(instruments):
        raise ValueError(f"Test spectra must cover exactly these instruments: {instruments}.")
    exact_lots(x, receipt["partitions"]["test"], "blinded test spectra")
    if set(x["preparation_id"]).intersection(receipt["training_preparations"]):
        raise ValueError("A test preparation ID was already used for development or calibration.")
    if len(instruments) == 2:
        exact_links(x.loc[x["instrument"].eq(instruments[0])], x.loc[x["instrument"].eq(instruments[1])], "paired test instruments")
    scaled = transform(bundle["scaler"], features(x, grid))
    results = []
    for name, model in bundle["learners"].items():
        scores = decision_score(name, model, scaled)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite test scores for {name}.")
        rows = x[SCAN_FIELDS[:3]].assign(
            model=name, score=scores, prediction=(scores > receipt["models"][name]["threshold"]).astype(int),
        )
        results.extend(rows.to_dict("records"))
    frozen_digest = sha256(data)
    report = {
        "frozen_sha256": frozen_digest,
        "frozen_receipt_sha256": sha256(json_bytes({**receipt, "frozen_sha256": frozen_digest})),
        "spectra_sha256": [sha256(path.read_bytes()) for path in args.spectra], "predictions": results,
    }
    encoded = json_bytes(report)
    write_new(args.out, encoded)
    print(f"Scored {len(x)} preparation–instrument pairs. SHA-256 {sha256(encoded)}")


def evaluate(args):
    receipt = json.loads(args.receipt.read_bytes())
    contents = args.scores.read_bytes()
    if sha256(contents) != args.expected_scores_sha256.lower():
        raise ValueError("The score file does not match the SHA-256 recorded before unblinding.")
    blind = json.loads(contents)
    if sha256(json_bytes(receipt)) != blind["frozen_receipt_sha256"]:
        raise ValueError("The receipt differs from the configuration used to score the test spectra.")
    labels = load_labels(args.labels)
    exact_lots(labels, receipt["partitions"]["test"], "unblinded test labels")
    complete_panel(labels, receipt["adulterants"])
    pred = pd.DataFrame(blind["predictions"])
    expected_instruments = {receipt["source_instrument"], receipt["target_instrument"]}
    expected_groups = {(model, instrument) for model in MODELS for instrument in expected_instruments}
    if set(pred[["model", "instrument"]].itertuples(index=False, name=None)) != expected_groups:
        raise ValueError("Missing or unexpected model and instrument combinations.")
    if not pred["prediction"].isin([0, 1]).all() or not np.isfinite(pred["score"].to_numpy(dtype=float)).all():
        raise ValueError("Invalid predictions in the blinded score artifact.")
    by_model, per_lot = {}, {}
    for (model, instrument), rows in pred.groupby(["model", "instrument"]):
        exact_links(rows, labels, f"{model} scores from {instrument}")
        expected_decisions = rows["score"].gt(receipt["models"][model]["threshold"])
        if not rows["prediction"].eq(expected_decisions).all():
            raise ValueError(f"Decisions do not match the frozen {model} threshold.")
        joined = rows.merge(labels, on=["lot_id", "preparation_id"], validate="one_to_one")
        success = lot_success(joined)
        per_lot[(model, instrument)] = success
        per_syrup = {}
        for (dose, syrup), group in joined.loc[joined["label"].eq(1)].groupby(["dose_pct", "adulterant"]):
            per_syrup.setdefault(syrup, {})[str(int(dose))] = {
                "detected": int(group["prediction"].sum()), "tested": len(group),
                "estimate": float(group["prediction"].mean()),
            }
        by_model.setdefault(model, {})[instrument] = {
            "endpoints": {name: rate_summary(success[name], receipt[name], receipt["alpha"]) for name in ENDPOINTS},
            "per_syrup_sensitivity": per_syrup,
        }
    primary = receipt["primary_model"]
    source, target = receipt["source_instrument"], receipt["target_instrument"]
    decision = all(value["meets_target"] for value in by_model[primary][target]["endpoints"].values())
    paired = {}
    if source != target:
        for model in MODELS:
            left, right = per_lot[(model, source)], per_lot[(model, target)]
            paired[model] = {
                name: {"target_minus_source": float(right[name].mean() - left[name].mean()),
                       "target_only": int((right[name] & ~left[name]).sum()),
                       "source_only": int((left[name] & ~right[name]).sum())}
                for name in ENDPOINTS
            }
    report = {
        "primary_model": primary, "target_instrument": target, "source_instrument": source,
        "go_decision": "pass" if decision else "fail", "one_sided_confidence": 1 - receipt["alpha"],
        "frozen_sha256": receipt["frozen_sha256"], "blinded_scores_sha256": sha256(contents),
        "test_labels_sha256": sha256(args.labels.read_bytes()), "models": by_model,
        "paired_instrument_changes": paired,
    }
    write_new(args.out, json_bytes(report))
    print(f"Primary decision for {primary} on {target}: {report['go_decision']}")


def fraction(value):
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("Use a fraction strictly between 0 and 1.")
    return parsed


def configure_targets(parser):
    parser.add_argument("--specificity", type=fraction, default=0.90)
    parser.add_argument("--sensitivity-5", type=fraction, default=0.80)
    parser.add_argument("--sensitivity-10", type=fraction, default=0.80)
    parser.add_argument("--alpha", type=fraction, default=0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="Estimate independent test lots needed to pass the gate.")
    configure_targets(planning)
    planning.add_argument("--assumed-specificity", type=fraction, default=0.96)
    planning.add_argument("--assumed-sensitivity-5", type=fraction, default=0.90)
    planning.add_argument("--assumed-sensitivity-10", type=fraction, default=0.92)
    planning.add_argument("--power", type=fraction, default=0.80)
    planning.add_argument("--max-lots", type=int, default=5000)
    fitting = commands.add_parser("freeze", help="Train on development lots and freeze thresholds on calibration pure controls.")
    configure_targets(fitting)
    fitting.add_argument("--partitions", required=True, type=Path)
    fitting.add_argument("--pilot-lots", type=Path, help="Reject final test lots listed in the pilot inventory.")
    fitting.add_argument("--labels", required=True, type=Path)
    fitting.add_argument("--spectra", required=True, type=Path, action="append")
    fitting.add_argument("--source-instrument", required=True)
    fitting.add_argument("--target-instrument")
    fitting.add_argument("--calibration-instrument")
    fitting.add_argument("--adulterants", nargs="+", required=True)
    fitting.add_argument("--min-nm", type=float, default=float(GRID_NM[0]))
    fitting.add_argument("--max-nm", type=float, default=float(GRID_NM[-1]))
    fitting.add_argument("--primary-model", default="XGBoost", choices=MODELS)
    fitting.add_argument("--out", required=True, type=Path)
    scoring = commands.add_parser("score", help="Score blinded spectra with a trusted, locally produced model bundle.")
    scoring.add_argument("--frozen", required=True, type=Path)
    scoring.add_argument("--spectra", required=True, type=Path, action="append")
    scoring.add_argument("--out", required=True, type=Path)
    evaluation = commands.add_parser("evaluate", help="Join opened labels to already frozen scores.")
    evaluation.add_argument("--receipt", required=True, type=Path)
    evaluation.add_argument("--scores", required=True, type=Path)
    evaluation.add_argument("--expected-scores-sha256", required=True)
    evaluation.add_argument("--labels", required=True, type=Path)
    evaluation.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            targets = (args.specificity, args.sensitivity_5, args.sensitivity_10)
            assumed = (args.assumed_specificity, args.assumed_sensitivity_5, args.assumed_sensitivity_10)
            if not 0 < args.max_lots <= 100000 or any(true <= minimum for true, minimum in zip(assumed, targets)):
                raise ValueError("Use positive --max-lots up to 100000 and assumptions above the corresponding targets.")
            print(json_bytes(sample_size_plan(targets, assumed, args.alpha, args.power, args.max_lots)).decode().rstrip())
            return
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            {"freeze": freeze, "score": score, "evaluate": evaluate}[args.command](args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
