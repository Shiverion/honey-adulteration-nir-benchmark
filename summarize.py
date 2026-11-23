"""Calculate the reported metrics and paired brand-level comparisons from predictions."""

import argparse
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


MODELS = ["PLS-DA", "SVM", "Random Forest", "XGBoost", "MLP", "1D-CNN", "Spectral Transformer"]
REFERENCE = "Selected classical"
CONTRASTS = [
    ("MLP", REFERENCE), ("1D-CNN", REFERENCE), ("Spectral Transformer", REFERENCE),
    ("1D-CNN", "MLP"), ("Spectral Transformer", "1D-CNN"),
]


def summarize_predictions(frame):
    y = frame["label"].to_numpy()
    pred = frame["prediction"].to_numpy()
    correct = y == pred
    n0, n1 = np.bincount(y, minlength=2)
    fold_aucs = [
        roc_auc_score(part["label"], part["decision"])
        for _, part in frame.groupby("fold") if part["label"].nunique() == 2
    ]
    return {
        "n_pure": n0, "n_adulterated": n1,
        "balanced_accuracy": balanced_accuracy_score(y, pred) if n0 and n1 else np.nan,
        "accuracy": correct.mean(),
        "macro_f1": f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0) if n0 and n1 else np.nan,
        "specificity": correct[y == 0].mean() if n0 else np.nan,
        "sensitivity": correct[y == 1].mean() if n1 else np.nan,
        "roc_auc_mean_within_fold": np.mean(fold_aucs) if fold_aucs else np.nan,
        "auc_folds": len(fold_aucs),
        "false_positives": int((pred[y == 0] == 1).sum()), "false_negatives": int((pred[y == 1] == 0).sum()),
    }


def group_counts(values, y, groups, n_groups):
    output = np.zeros((n_groups, 2), dtype=float)
    np.add.at(output, (groups, y), values)
    return output


def holm(pvalues):
    rank = np.argsort(pvalues)
    ordered = np.asarray(pvalues)[rank]
    corrected = np.empty(len(rank))
    corrected[rank] = np.minimum(1, np.maximum.accumulate(ordered * np.arange(len(rank), 0, -1)))
    return corrected


def brand_inference(clean):
    scores = clean.pivot(index=["fold", "sample_id", "label"], columns="model", values="prediction")
    if scores.isna().any().any():
        raise ValueError("Paired comparison is missing a model prediction.")
    y = scores.index.get_level_values("label").to_numpy()
    groups, indices = np.unique(scores.index.get_level_values("fold"), return_inverse=True)
    counts = group_counts(1, y, indices, len(groups))
    draws = np.random.default_rng(741).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=20000)
    denominators = draws @ counts
    valid = (denominators > 0).all(axis=1)
    draws, denominators = draws[valid], denominators[valid]
    signs = np.array(list(product((-1, 1), repeat=len(groups))))

    def effect(a, b=None):
        difference = (scores[a].to_numpy() == y).astype(int)
        if b is not None:
            difference -= scores[b].to_numpy() == y
        correct = group_counts(difference, y, indices, len(groups))
        contributions = (correct / counts.sum(axis=0)).mean(axis=1)
        drawn = ((draws @ correct) / denominators).mean(axis=1)
        lo, hi = np.quantile(drawn, [0.025, 0.975])
        observed = contributions.sum()
        return observed, lo, hi, contributions

    estimates = []
    for model in MODELS + [REFERENCE]:
        observed, lo, hi, _ = effect(model)
        estimates.append({"model": model, "balanced_accuracy": observed, "ci95_low": lo, "ci95_high": hi})
    comparisons = []
    for model, baseline in CONTRASTS:
        observed, lo, hi, contributions = effect(model, baseline)
        null = signs @ contributions
        comparisons.append({
            "model": model, "baseline": baseline, "difference": observed,
            "ci95_low": lo, "ci95_high": hi,
            "p_one_sided": float(np.mean(null >= observed - 1e-12)),
            "nonzero_brand_contrasts": int(np.count_nonzero(np.abs(contributions) > 1e-12)),
            "min_attainable_one_sided_p": float(np.mean(null >= null.max() - 1e-12)),
            "brand_clusters": len(groups), "valid_bootstrap_draws": len(draws),
        })
    comparisons = pd.DataFrame(comparisons)
    comparisons["p_holm"] = holm(comparisons["p_one_sided"].to_numpy())
    return pd.DataFrame(estimates), comparisons


def rows_for(frame, by):
    return pd.DataFrame([
        {**dict(zip(by, keys)), **summarize_predictions(part)}
        for keys, part in frame.groupby(by, sort=False)
    ])


def plot(metrics, uncertainty, robustness, comparisons, efficiency, destination):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    y = np.arange(len(MODELS))
    cohort = metrics.loc[metrics["scenario"].eq("clean")].pivot(index="model", columns="protocol", values="balanced_accuracy")
    inside = uncertainty.set_index("model").loc[MODELS]
    axis = axes[0, 0]
    axis.hlines(y - 0.14, 100 * inside["ci95_low"], 100 * inside["ci95_high"], color="#174566")
    axis.scatter(100 * cohort.loc[MODELS, "held_out_brand"], y - 0.14, color="#174566", label="Held-out brand")
    axis.scatter(100 * cohort.loc[MODELS, "cross_dataset"], y + 0.14, marker="D", color="#A34D29", label="External dataset")
    axis.axvline(50, color="0.55", ls=":")
    axis.set(yticks=y, yticklabels=MODELS, xlim=(0, 102), xlabel="Balanced accuracy (%)", title="Classification and transfer")
    axis.invert_yaxis()
    axis.legend(loc="lower left", frameon=False)

    axis = axes[0, 1]
    rows = comparisons.iloc[::-1].reset_index(drop=True)
    for i, row in rows.iterrows():
        axis.hlines(i, 100 * row["ci95_low"], 100 * row["ci95_high"], color="#174566")
        axis.scatter(100 * row["difference"], i, color="#174566", zorder=3)
    axis.axvline(0, color="0.55", ls=":")
    labels = [f"{row.model} vs\n{row.baseline}" for row in rows.itertuples()]
    axis.set(yticks=np.arange(len(rows)), yticklabels=labels, xlabel="Balanced accuracy difference (pp)", title="Paired neural comparisons")

    axis = axes[1, 0]
    changes = robustness.pivot(index="model", columns="scenario", values="delta_from_clean").loc[MODELS]
    scenario_names = {"noise_1pct": "1% noise", "noise_5pct": "5% noise", "tilt_10pct": "10% tilt", "shift_5nm": "5 nm shift"}
    palette = ["#2B687A", "#A34D29", "#623E82", "#6A7021"]
    for i, (key, label) in enumerate(scenario_names.items()):
        axis.scatter(100 * changes[key], y + (i - 1.5) * 0.13, label=label, color=palette[i])
    axis.axvline(0, color="0.55", ls=":")
    axis.set(yticks=y, yticklabels=MODELS, xlabel="Change in balanced accuracy (pp)", title="Sensitivity to spectral perturbations")
    axis.invert_yaxis()
    axis.legend(frameon=False, ncols=2, fontsize=8)

    axis = axes[1, 1]
    rows = efficiency.set_index("model").loc[MODELS]
    axis.scatter(rows["final_fit_s"], y - 0.14, color="#174566", label="Refit")
    axis.scatter(rows["inner_search_s"], y + 0.14, color="#A34D29", marker="D", label="Inner search")
    axis.set(yticks=y, yticklabels=MODELS, xlabel="Median seconds per outer fold (log scale)",
             xscale="log", title="Single-thread CPU training time")
    axis.invert_yaxis()
    axis.legend(frameon=False)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    pred = pd.read_csv(args.out / "predictions.csv")
    timings = pd.read_csv(args.out / "timings.csv")
    metrics = rows_for(pred, ["protocol", "scenario", "model"])
    clean = pred.loc[pred["scenario"].eq("clean")]
    grouped = clean.loc[clean["protocol"].eq("held_out_brand")]
    external = clean.loc[clean["protocol"].eq("cross_dataset")]
    uncertainty, comparisons = brand_inference(grouped)
    robustness = metrics.loc[(metrics["protocol"] == "held_out_brand") & (metrics["scenario"] != "clean")].copy()
    baseline = metrics.loc[(metrics["protocol"] == "held_out_brand") & (metrics["scenario"] == "clean")].set_index("model")
    robustness["delta_from_clean"] = robustness["balanced_accuracy"] - robustness["model"].map(baseline["balanced_accuracy"])
    by_brand = rows_for(grouped, ["fold", "model"])
    by_dose = rows_for(grouped, ["nominal_pct", "model"])
    by_syrup = rows_for(external, ["adulterant", "model"])
    timing_stats = timings.loc[timings["protocol"].eq("held_out_brand")].groupby("model", sort=False)
    efficiency = timing_stats.median(numeric_only=True).reset_index()
    for name, table in (
        ("metrics", metrics), ("brand_uncertainty", uncertainty), ("comparisons", comparisons),
        ("robustness", robustness), ("by_brand", by_brand), ("by_dose", by_dose),
        ("external_by_adulterant", by_syrup), ("efficiency", efficiency),
    ):
        table.to_csv(args.out / f"{name}.csv", index=False)
    plot(metrics, uncertainty, robustness, comparisons, efficiency, args.out / "benchmark.png")


if __name__ == "__main__":
    main()
