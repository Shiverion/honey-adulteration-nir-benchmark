"""Run grouped model selection, held-out prediction and measurement on one CPU thread."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from datasets import GRID_NM, SOURCES, get_file, load_auckland, load_nirpy
from models import CLASSICAL, MODELS, SEARCH, decision_score, fit_model, footprint


SEED = 20261123
REFERENCE = "Selected classical"
STRESSES = ("clean", "noise_1pct", "noise_5pct", "tilt_10pct", "shift_5nm")


def snv(x):
    scale = x.std(axis=1, keepdims=True)
    if (scale < 1e-12).any():
        raise ValueError("SNV cannot normalize a constant spectrum.")
    return (x - x.mean(axis=1, keepdims=True)) / scale


def transform(scaler, x):
    return scaler.transform(snv(x)).astype(np.float32)


def validate_split(y, groups, train, test):
    if np.intersect1d(groups[train], groups[test]).size or set(y[train]) != {0, 1}:
        raise ValueError("A split leaks groups or lacks a training class.")


def prepare_inner(x, y, groups, n_splits=3):
    splits = []
    # A held-out brand may contain only positives. Tune on pooled inner OOF predictions.
    for train, valid in GroupKFold(n_splits=n_splits).split(x, y, groups):
        validate_split(y, groups, train, valid)
        scaler = StandardScaler().fit(snv(x[train]))
        splits.append((transform(scaler, x[train]), y[train], transform(scaler, x[valid]), valid))
    return splits


def tune(name, y, splits, seed):
    start = time.perf_counter()
    scores = []
    for params in SEARCH[name]:
        decisions = np.full(len(y), np.nan)
        for number, (train, labels, valid, rows) in enumerate(splits):
            model = fit_model(name, params, train, labels, seed + number)
            decisions[rows] = decision_score(name, model, valid)
        if not np.isfinite(decisions).all():
            raise ValueError("Some inner predictions are missing or non-finite.")
        scores.append(balanced_accuracy_score(y, decisions > 0))
    index = int(np.argmax(scores))
    return index, scores, time.perf_counter() - start


def stresses(x, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=x.shape) * x.std(axis=1, keepdims=True)
    tilt = np.linspace(-0.5, 0.5, x.shape[1]) * np.ptp(x, axis=1, keepdims=True)
    offsets = rng.choice([-5, 5], size=len(x))
    shifted = np.stack([np.interp(GRID_NM + offset, GRID_NM, row) for row, offset in zip(x, offsets)])
    return dict(zip(STRESSES, (x, x + 0.01 * noise, x + 0.05 * noise, x + 0.10 * tilt, shifted)))


def timed_prediction(name, model, scaler, x):
    measurements = {}
    for n, repeats in ((1, 15), (128, 10)):
        batch = x[np.arange(n) % len(x)]
        for _ in range(3):
            decision_score(name, model, transform(scaler, batch))
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            decision_score(name, model, transform(scaler, batch))
            elapsed.append((time.perf_counter_ns() - start) / 1000 / n)
        measurements[f"batch_{n}_us_per_spectrum"] = float(np.median(elapsed))
    return measurements


def evaluate(source, target, out):
    groups = source.meta["group"].to_numpy()
    jobs = []
    for train, test in LeaveOneGroupOut().split(source.x, source.y, groups):
        validate_split(source.y, groups, train, test)
        jobs.append(("held_out_brand", groups[test][0], train, source.x[test], source.meta.iloc[test]))
    jobs.append(("cross_dataset", "Argentina", np.arange(len(source.x)), target.x, target.meta))

    predictions, timing, candidates, splits_log = [], [], [], []
    for fold, (protocol, test_name, idx, test_x, test_meta) in enumerate(jobs):
        seed = SEED + fold * 100
        train_x, train_y, train_groups = source.x[idx], source.y[idx], groups[idx]
        inner_start = time.perf_counter()
        inner = prepare_inner(train_x, train_y, train_groups)
        shared_inner_prep_s = time.perf_counter() - inner_start
        splits_log.append({
            "protocol": protocol, "fold": test_name, "seed": seed,
            "outer_source_training": sorted(set(train_groups)),
            "inner_validation": [sorted(set(train_groups[part[3]])) for part in inner],
            "test_rows": len(test_x),
        })
        start = time.perf_counter()
        scaler = StandardScaler().fit(snv(train_x))
        train = transform(scaler, train_x)
        preprocessing_s = time.perf_counter() - start
        stress_inputs = stresses(test_x, seed) if protocol == "held_out_brand" else {"clean": test_x}
        test = {scenario: transform(scaler, passed) for scenario, passed in stress_inputs.items()}
        fold_models = {}
        for name in MODELS:
            index, scores, tuning_s = tune(name, train_y, inner, seed)
            params = SEARCH[name][index]
            for candidate, value in enumerate(scores):
                candidates.append({
                    "protocol": protocol, "fold": test_name, "model": name, "candidate": candidate,
                    "params": json.dumps(SEARCH[name][candidate], sort_keys=True), "inner_balanced_accuracy": value,
                    "selected": int(index == candidate),
                })
            start = time.perf_counter()
            model = fit_model(name, params, train, train_y, seed + 50)
            fit_s = time.perf_counter() - start
            bytes_, parameters = footprint(name, model)
            timing.append({
                "protocol": protocol, "fold": test_name, "model": name, "train_rows": len(train),
                "final_fit_s": fit_s, "inner_search_s": tuning_s, "final_preprocessing_s": preprocessing_s,
                "shared_inner_preprocessing_s": shared_inner_prep_s, "serialized_bytes": bytes_,
                "trainable_parameters": parameters, **timed_prediction(name, model, scaler, test_x),
            })
            frames = []
            for scenario, features in test.items():
                values = decision_score(name, model, features)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite {name} predictions in {test_name}.")
                frames.append(test_meta.assign(
                    protocol=protocol, fold=test_name, scenario=scenario, model=name,
                    selected_classical="", decision=values, prediction=(values > 0).astype(int),
                ))
            fold_models[name] = (scores[index], pd.concat(frames, ignore_index=True))
            predictions.append(fold_models[name][1])
            print(f"{test_name} {name}: best inner BA={scores[index]:.3f}, search={tuning_s:.2f}s", flush=True)
        preferred = max(CLASSICAL, key=lambda name: fold_models[name][0])
        predictions.append(fold_models[preferred][1].assign(model=REFERENCE, selected_classical=preferred))

    pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "timings.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "tuning.csv", index=False)
    (out / "splits.json").write_text(json.dumps(splits_log, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--download", action="store_true", help="Fetch missing, checksum-pinned sources.")
    args = parser.parse_args()
    source = load_auckland(get_file(args.data, "auckland", args.download))
    target = load_nirpy(get_file(args.data, "nirpy", args.download))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run = {
        "python": platform.python_version(), "os": platform.platform(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()), "threads": 1,
        "packages": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "matplotlib", "threadpoolctl",
        )},
        "seed": SEED, "wavelength_grid_nm": GRID_NM.tolist(),
        "datasets": {},
    }
    for key, data in (("auckland", source), ("nirpy", target)):
        run["datasets"][key] = {
            **data.audit, "sha256": SOURCES[key][2], "observations": len(data.x),
            "class_counts": {str(i): int(np.sum(data.y == i)) for i in (0, 1)},
        }
        data.meta.to_csv(args.out / f"{key}_observations.csv", index=False)
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    with threadpool_limits(limits=1):
        evaluate(source, target, args.out)


if __name__ == "__main__":
    main()
