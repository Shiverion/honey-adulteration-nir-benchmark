"""Pre-specified endpoints and exact binomial calculations for independent lots."""

import numpy as np
import pandas as pd
from scipy.stats import beta, binom


ENDPOINTS = ("specificity", "sensitivity_5pct", "sensitivity_10pct")


def threshold_for_specificity(pure_scores, target):
    scores = np.sort(np.asarray(pure_scores, dtype=float))
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Threshold calibration requires finite scores for pure controls.")
    rank = np.searchsorted(np.arange(1, len(scores) + 1) / len(scores), target)
    return float(scores[rank])


def rate_summary(success, target, alpha):
    count, total = int(np.count_nonzero(success)), len(success)
    lower = float(beta.ppf(alpha, count, total - count + 1)) if count else 0.0
    return {
        "passed_lots": count, "total_lots": total, "estimate": count / total,
        "one_sided_lower_bound": lower, "target": target, "meets_target": lower >= target,
    }


def complete_panel(labels, adulterants):
    expected = {(0.0, "none")} | {(dose, syrup) for dose in (5.0, 10.0) for syrup in adulterants}
    for lot, rows in labels.groupby("lot_id", sort=True):
        observed = set(zip(rows["dose_pct"], rows["adulterant"]))
        if observed != expected or len(rows) != len(expected):
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(f"Incomplete or repeated test panel for {lot!r}. Missing: {missing}. Unexpected: {extra}.")


def lot_success(rows):
    success = pd.DataFrame(index=sorted(rows["lot_id"].unique()))
    for name, dose in zip(ENDPOINTS, (0.0, 5.0, 10.0)):
        subset = rows.loc[rows["dose_pct"].eq(dose)]
        correct = subset["prediction"].eq(int(dose > 0))
        success[name] = correct.groupby(subset["lot_id"]).all()
    return success


def sample_size_plan(targets, assumed, alpha, power, max_lots):
    sizes = np.arange(1, max_lots + 1)
    criteria = []
    for name, minimum, alternative in zip(ENDPOINTS, targets, assumed):
        # Check the one-sided exact test directly. The extra check handles exact ties.
        critical = binom.isf(alpha, sizes, minimum).astype(int) + 1
        previous_passes = binom.sf(critical - 2, sizes, minimum) <= alpha
        critical -= previous_passes
        endpoint_power = binom.sf(critical - 1, sizes, alternative)
        possible = np.flatnonzero(critical <= sizes)
        criteria.append((name, critical, endpoint_power, int(possible[0] + 1) if len(possible) else None))
    power_floor = np.maximum(0, 1 - sum(1 - endpoint[2] for endpoint in criteria))
    eligible = np.flatnonzero(power_floor >= power)
    i = int(eligible[0]) if len(eligible) else None
    return {
        "test_lots": i + 1 if i is not None else None,
        "searched_up_to_lots": max_lots,
        "joint_power_lower_bound": float(power_floor[i]) if i is not None else None,
        "desired_joint_power": power,
        "one_sided_confidence": 1 - alpha,
        "endpoints": {
            name: {
                "target": target, "assumed_true_rate": alternative,
                "first_lot_count_where_passing_is_possible": first,
                "minimum_successes_at_planned_size": int(critical[i]) if i is not None else None,
                "power_at_planned_size": float(powers[i]) if i is not None else None,
            }
            for (name, critical, powers, first), target, alternative in zip(criteria, targets, assumed)
        },
    }
