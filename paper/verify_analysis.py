"""Recompute the manuscript's key rates from observation-level predictions."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify_benchmark(folder):
    pred = pd.read_csv(folder / "predictions.csv")
    metrics = pd.read_csv(folder / "metrics.csv")
    for row in metrics.itertuples():
        data = pred.loc[pred["model"].eq(row.model) & pred["protocol"].eq(row.protocol) & pred["scenario"].eq(row.scenario)]
        assert data["sample_id"].is_unique
        correct = data["prediction"].eq(data["label"])
        assert np.isclose(correct.groupby(data["label"]).mean().mean(), row.balanced_accuracy)
    clean = pred.loc[pred["protocol"].eq("held_out_brand") & pred["scenario"].eq("clean")]
    decisions = clean.pivot(index=["sample_id", "fold", "label"], columns="model", values="prediction").reset_index()
    label_totals = decisions["label"].value_counts()
    weights = decisions["label"].map(label_totals).rdiv(.5)
    comparisons = pd.read_csv(folder / "comparisons.csv")
    for row in comparisons.itertuples():
        difference = decisions[row.model].eq(decisions["label"]).astype(int) - decisions[row.baseline].eq(decisions["label"]).astype(int)
        by_brand = (difference * weights).groupby(decisions["fold"]).sum().to_numpy()
        observed = by_brand.sum()
        null = np.array(list(product([-1, 1], repeat=len(by_brand)))) @ by_brand
        assert np.isclose(observed, row.difference)
        assert np.isclose(np.mean(null >= observed - 1e-12), row.p_one_sided)
        assert np.count_nonzero(np.abs(by_brand) > 1e-12) == row.nonzero_brand_contrasts
        assert np.isclose(np.mean(null >= null.max() - 1e-12), row.min_attainable_one_sided_p)
    order = np.argsort(comparisons["p_one_sided"])
    ordered = comparisons["p_one_sided"].to_numpy()[order]
    corrected = np.minimum(1, np.maximum.accumulate(ordered * np.arange(len(order), 0, -1)))
    assert np.allclose(corrected, comparisons["p_holm"].to_numpy()[order])


def verify_transfer(folder):
    source = pd.read_csv(folder / "auckland_predictions.csv")
    external = pd.read_csv(folder / "padua_predictions.csv")
    manifest = json.loads((folder / "split_manifest.json").read_bytes())
    assert len(manifest) == 15
    for entry in manifest:
        dev, cal = set(entry["development_brands"]), set(entry["calibration_brands"])
        assert not dev.intersection(cal) and entry["test"] not in dev | cal
        peers = [row for row in manifest if row["test"] == entry["test"]]
        assert all(set(row["development_brands"]) == dev and set(row["calibration_brands"]) == cal for row in peers)
        assert all(model["calibration_specificity"] >= .9 for model in entry["models"].values())
    assert not source.duplicated(["model", "window", "sample_id"]).any()
    assert not external.duplicated(["model", "representation", "SampleID"]).any()
    assert source["flagged"].eq(source["score"].gt(source["threshold"])).all()
    assert external["flagged"].eq(external["score"].gt(external["threshold"])).all()
    for row in pd.read_csv(folder / "auckland_metrics.csv").itertuples():
        data = source.loc[source["window"].eq(row.window) & source["model"].eq(row.model)]
        for dose, metric in ((0, "specificity"), (5, "sensitivity_5pct"), (10, "sensitivity_10pct")):
            subset = data.loc[data["nominal_pct"].eq(dose)]
            correct = subset["flagged"].eq(subset["label"])
            assert len(correct) == getattr(row, f"{metric}_total")
            assert correct.sum() == getattr(row, f"{metric}_correct")
            assert np.isclose(correct.mean(), getattr(row, metric))
    for row in pd.read_csv(folder / "padua_flag_rates.csv").itertuples():
        data = external.loc[external["representation"].eq(row.representation) & external["model"].eq(row.model)]
        assert len(data) == row.samples == 80
        assert data["flagged"].sum() == row.flags
        assert np.isclose(data["flagged"].mean(), row.flag_rate)
    support = pd.read_csv(folder / "feature_support.csv")
    assert not support.duplicated(["cohort", "observation_id"]).any()
    assert support["features"].eq(48).all()
    assert support["features_outside_development_range"].between(0, support["features"]).all()
    padua_ids = set(external["SampleID"].astype(str))
    for cohort in external["representation"].unique():
        assert set(support.loc[support["cohort"].eq(cohort), "observation_id"]) == padua_ids
    frozen = next(entry for entry in manifest if entry["test"] == "Padua")
    controls = source.loc[source["group"].isin(frozen["calibration_brands"]) & source["label"].eq(0), "sample_id"]
    assert set(support.loc[support["cohort"].eq("auckland_calibration_pure"), "observation_id"]) == set(controls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=Path("results"), type=Path)
    parser.add_argument("--transfer", default=Path("results/padua_transfer"), type=Path)
    args = parser.parse_args()
    verify_benchmark(args.benchmark)
    verify_transfer(args.transfer)
    print("Verified benchmark rates, permutation tests and their resolution, calibrated source rates, and Padua outputs.")
