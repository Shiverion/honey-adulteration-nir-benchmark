"""Export the manuscript's benchmark figure from the saved analysis tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parent.parent
results = root / "results"
figure_dir = root / "paper" / "figures"
figure_dir.mkdir(exist_ok=True)
summary = pd.read_csv(results / "metrics.csv").query("scenario == 'clean'")
points = summary.pivot(index="model", columns="protocol", values="balanced_accuracy")
intervals = pd.read_csv(results / "brand_uncertainty.csv").set_index("model")
models = ["PLS-DA", "SVM", "Random Forest", "XGBoost", "MLP", "1D-CNN", "Spectral Transformer", "Selected classical"]
y = np.arange(len(models))
fig, ax = plt.subplots(figsize=(6.55, 3.45), layout="constrained")
ax.axvline(50, color="0.5", linestyle="--", linewidth=1)
ax.hlines(y - .13, 100 * intervals.loc[models, "ci95_low"], 100 * intervals.loc[models, "ci95_high"], color="#215A7A")
ax.scatter(100 * points.loc[models, "held_out_brand"], y - .13, color="#215A7A", s=30, label="Held-out Auckland brands")
ax.scatter(100 * points.loc[models, "cross_dataset"], y + .13, color="#A64B29", marker="D", s=23, label="Argentina transfer")
ax.set(xlim=(30, 100), yticks=y, yticklabels=models, xlabel="Balanced accuracy (%)")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=False, fontsize=8.5)
fig.savefig(figure_dir / "benchmark_results.png", dpi=350)
fig.savefig(figure_dir / "benchmark_results.pdf")
