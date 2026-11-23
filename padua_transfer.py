"""Auckland-only threshold calibration and frozen scoring of Padua FOSS spectra."""

import argparse
from importlib.metadata import version
import pickle
from pathlib import Path

from benchmark import SEED, decision_score, fit_model, prepare_inner, snv, transform, tune

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from datasets import GRID_NM, SOURCES, get_file, load_auckland, resample
from followup import MODELS, json_bytes, sha256, write_new
from followup_stats import threshold_for_specificity
from models import SEARCH


REFERENCE = "780-1000"
MATCHED = "850-1000"
GRIDS = {REFERENCE: GRID_NM, MATCHED: np.linspace(850.0, 1000.0, len(GRID_NM))}
TARGET_SPECIFICITY = 0.90
DOSES = (0, 5, 10)
RATE_NAMES = ("specificity", "sensitivity_5pct", "sensitivity_10pct")


def read_padua(path):
    with path.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=True)
        try:
            records = book["FOSS_DS-2500_spectra"].values
            header, *bands = next(records)
            if header != "SampleID":
                raise ValueError("Padua FOSS sheet is missing its sample ID header.")
            ids, *values = zip(*records)
            quality = book["Chemical_Analysis"].values
            names = [name.strip() for name in next(quality)]
            metadata = pd.DataFrame(quality, columns=names)[["SampleID", "Origin", "Region"]]
        finally:
            book.close()
    if len(ids) != 80 or len(set(ids)) != len(ids) or set(ids) != set(metadata["SampleID"]):
        raise ValueError("The 80 FOSS IDs must match the workbook's chemical analysis sheet.")
    absorbance = resample(np.array(values, dtype=float).T, np.array(bands, dtype=float), GRIDS[MATCHED])
    features = {"reflectance_proxy": 10 ** (-absorbance), "as_recorded_absorbance": absorbance}
    return metadata.set_index("SampleID").loc[list(ids)].reset_index(), features


def source_splits(meta):
    groups = meta["group"].to_numpy()
    all_brands = np.unique(groups)
    pure_brands = np.unique(groups[meta["label"].eq(0)])
    rng = np.random.default_rng(SEED)
    for fold, test_brand in enumerate([*all_brands, "Padua"], start=1):
        choices = pure_brands[pure_brands != test_brand]
        calibration = sorted(rng.choice(choices, size=2, replace=False))
        development = sorted(set(all_brands) - set(calibration) - {test_brand})
        yield fold, test_brand, development, calibration


def fit_source(x, y, groups, development, calibration, seed):
    dev, cal = np.isin(groups, development), np.isin(groups, calibration)
    pure = cal & (y == 0)
    if (dev & cal).any() or len(np.unique(groups[dev & (y == 0)])) < 2 or not pure.any():
        raise ValueError("Invalid source-only development and calibration split.")
    # Each inner fit leaves out a development brand and retains at least one pure brand.
    inner = prepare_inner(x[dev], y[dev], groups[dev], n_splits=len(development))
    scaler = StandardScaler().fit(snv(x[dev]))
    training = transform(scaler, x[dev])
    controls = transform(scaler, x[pure])
    models, fit_log = {}, {}
    for name in MODELS:
        i, candidates, _ = tune(name, y[dev], inner, seed)
        params = SEARCH[name][i]
        model = fit_model(name, params, training, y[dev], seed + 50)
        scores = decision_score(name, model, controls)
        threshold = threshold_for_specificity(scores, TARGET_SPECIFICITY)
        models[name] = (model, threshold)
        fit_log[name] = {"params": params, "inner_balanced_accuracy": candidates, "threshold": threshold,
                         "calibration_pure_count": len(scores), "calibration_specificity": float(np.mean(scores <= threshold))}
    return scaler, models, fit_log


def score_frame(models, scaler, x, meta, **columns):
    prepared = transform(scaler, x)
    frames = []
    for name, (model, threshold) in models.items():
        scores = decision_score(name, model, prepared)
        if not np.isfinite(scores).all():
            raise ValueError(f"{name} produced non-finite scores.")
        frames.append(meta.assign(model=name, score=scores, threshold=threshold, flagged=(scores > threshold).astype(int), **columns))
    return pd.concat(frames, ignore_index=True)


def source_analysis(source_path, out):
    splits, predictions = [], []
    for name, grid in GRIDS.items():
        source = load_auckland(source_path, grid)
        groups = source.meta["group"].to_numpy()
        for fold, test_brand, development, calibration in source_splits(source.meta):
            if test_brand == "Padua" and name != MATCHED:
                continue
            if test_brand in development or test_brand in calibration:
                raise ValueError("The held-out brand was included in a fitting role.")
            seed = SEED + 100 * fold
            scaler, models, fit_log = fit_source(source.x, source.y, groups, development, calibration, seed)
            log = {"window": name, "test": test_brand, "development_brands": development,
                   "calibration_brands": calibration, "seed": seed, "models": fit_log}
            splits.append(log)
            if test_brand == "Padua":
                payload = {"scaler": scaler, "models": models, "grid_nm": grid, "log": log,
                           "auckland_sha256": SOURCES["auckland"][2]}
                write_new(out / "frozen_auckland.pkl", pickle.dumps(payload, protocol=5))
            else:
                mask = groups == test_brand
                predictions.append(score_frame(models, scaler, source.x[mask], source.meta.loc[mask], window=name))
            print(f"{name} | {test_brand} scored or frozen", flush=True)
    write_new(out / "split_manifest.json", json_bytes(splits))
    return pd.concat(predictions, ignore_index=True)


def source_rates(pred):
    results = []
    for (model, window), rows in pred.groupby(["model", "window"], sort=False):
        correct = rows["flagged"].eq(rows["label"])
        rates = {}
        for name, dose in zip(RATE_NAMES, DOSES):
            keep = rows["nominal_pct"].eq(dose)
            rates[name] = float(correct[keep].mean())
            rates[f"{name}_correct"] = int(correct[keep].sum())
            rates[f"{name}_total"] = int(keep.sum())
        rates["balanced_accuracy"] = (rates["specificity"] + correct[rows["label"].eq(1)].mean()) / 2
        results.append({"model": model, "window": window, **rates})
    return pd.DataFrame(results)


def paired_dose_differences(pred):
    results = []
    indices = ["group", "sample_id", "nominal_pct", "label"]
    for name, rows in pred.groupby("model", sort=False):
        paired = rows.pivot(index=indices, columns="window", values="flagged").reset_index()
        if paired[list(GRIDS)].isna().any().any():
            raise ValueError("Window comparisons require exactly paired predictions.")
        paired["gain"] = paired[MATCHED].eq(paired["label"]).astype(int) - paired[REFERENCE].eq(paired["label"]).astype(int)
        groups = sorted(paired["group"].unique())
        draws = np.random.default_rng(741).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=20000)
        for endpoint, dose in zip(RATE_NAMES, DOSES):
            subset = paired.loc[paired["nominal_pct"].eq(dose)]
            counts = subset.groupby("group")["gain"].agg(["sum", "size"]).reindex(groups, fill_value=0)
            totals = draws @ counts["size"]
            valid = totals > 0
            differences = (draws[valid] @ counts["sum"]) / totals[valid]
            lo, hi = np.quantile(differences, [0.025, 0.975])
            results.append({"model": name, "endpoint": endpoint, "matched_minus_reference": float(subset["gain"].mean()),
                            "ci95_low": lo, "ci95_high": hi, "supporting_brands": int(counts["size"].gt(0).sum())})
    return pd.DataFrame(results)


def feature_support(source_path, frozen_log, padua_meta, input_spaces):
    """Count SNV features beyond Auckland development minima or maxima."""
    source = load_auckland(source_path, GRIDS[MATCHED])
    brands = source.meta["group"]
    development = snv(source.x[brands.isin(frozen_log["development_brands"])])
    low, high = development.min(axis=0), development.max(axis=0)
    pure = brands.isin(frozen_log["calibration_brands"]) & (source.y == 0)
    cohorts = [("auckland_calibration_pure", source.meta.loc[pure, "sample_id"], source.x[pure])]
    cohorts.extend((name, padua_meta["SampleID"], spectra) for name, spectra in input_spaces.items())
    rows = []
    for name, ids, spectra in cohorts:
        normalized = snv(spectra)
        outside = np.sum((normalized < low) | (normalized > high), axis=1)
        rows.append(pd.DataFrame({"cohort": name, "observation_id": ids.to_numpy(),
                                  "features": development.shape[1], "features_outside_development_range": outside}))
    return pd.concat(rows, ignore_index=True)


def plot_transfer(rates, target, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.1), layout="constrained")
    colors = {"SVM": "#215A7A", "XGBoost": "#A64B29", "MLP": "#58712A"}
    for i, name in enumerate(MODELS):
        r = rates.loc[rates["model"].eq(name)].set_index("window")
        for offset, field in enumerate(RATE_NAMES):
            y = i * 3.8 + offset
            values = [100 * r.loc[w, field] for w in GRIDS]
            left.plot(values, [y, y], color=colors[name], alpha=.45)
            for j, marker in enumerate(("o", "D")):
                left.scatter(values[j], y, marker=marker, facecolors="white" if j == 0 else colors[name], edgecolors=colors[name])
        report = target.loc[target["model"].eq(name)].set_index("representation")
        right.scatter(100 * report.loc["reflectance_proxy", "flag_rate"], i - .12, marker="D", color=colors[name])
        right.scatter(100 * report.loc["as_recorded_absorbance", "flag_rate"], i + .12, marker="o", facecolors="white", edgecolors=colors[name])
    for axis in (left, right):
        axis.set_xlim(-3, 103)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=.2)
        axis.invert_yaxis()
    left.set(yticks=[i * 3.8 + j for i in range(len(MODELS)) for j in range(3)],
             yticklabels=[f"{name} · {label}" for name in MODELS for label in ("pure", "5%", "10%")],
             xlabel="Correctly classified (%)", title="Held-out Auckland brands")
    right.set(yticks=range(len(MODELS)), yticklabels=MODELS, xlabel="Samples flagged (%)", title="Padua FOSS spectra")
    from matplotlib.lines import Line2D
    for axis, labels in ((left, ("780–1000 nm", "850–1000 nm")), (right, ("As-recorded absorbance", "Reflectance proxy"))):
        axis.legend(handles=[Line2D([], [], marker=marker, color="black", linestyle="none", markerfacecolor=fill, label=label)
                             for marker, fill, label in zip(("o", "D"), ("white", "black"), labels)],
                    loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False, fontsize=8)
    fig.savefig(out / "padua_transfer.png", dpi=320)
    fig.savefig(out / "padua_transfer.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=Path("data"), type=Path)
    parser.add_argument("--out", default=Path("results/padua_transfer"), type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    source_path = get_file(args.data, "auckland", args.download)
    args.out.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        source = source_analysis(source_path, args.out)
        source.to_csv(args.out / "auckland_predictions.csv", index=False)
        frozen = (args.out / "frozen_auckland.pkl").read_bytes()
        bundle = pickle.loads(frozen)
        metadata, input_spaces = read_padua(get_file(args.data, "padua", args.download))
        external = pd.concat([score_frame(bundle["models"], bundle["scaler"], x, metadata, representation=name)
                              for name, x in input_spaces.items()], ignore_index=True)
    external.to_csv(args.out / "padua_predictions.csv", index=False)
    rates = source_rates(source)
    rates.to_csv(args.out / "auckland_metrics.csv", index=False)
    paired_dose_differences(source).to_csv(args.out / "paired_differences.csv", index=False)
    target = external.groupby(["model", "representation"], sort=False)["flagged"].agg(flags="sum", samples="size", flag_rate="mean").reset_index()
    target.to_csv(args.out / "padua_flag_rates.csv", index=False)
    feature_support(source_path, bundle["log"], metadata, input_spaces).to_csv(args.out / "feature_support.csv", index=False)
    write_new(args.out / "run.json", json_bytes({
        "seed": SEED, "frozen_sha256": sha256(frozen), "calibration_target": TARGET_SPECIFICITY,
        "grids_nm": {name: grid.tolist() for name, grid in GRIDS.items()},
        "source_sha256": {name: SOURCES[name][2] for name in ("padua", "auckland")},
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "openpyxl")},
    }))
    plot_transfer(rates, target, args.out)


if __name__ == "__main__":
    main()
