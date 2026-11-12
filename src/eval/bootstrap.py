"""Paired bootstrap confidence intervals over per-question metric differences."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(
    a,
    b,
    n_boot: int = 1000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> dict:
    """Mean of a-b with a percentile interval from paired resampling."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    n = int(a.size)
    if n == 0:
        return {"mean_diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "n_boot": n_boot}
    diff = a - b
    mean_diff = float(diff.mean())
    if np.all(diff == diff[0]):
        return {
            "mean_diff": mean_diff,
            "lo": mean_diff,
            "hi": mean_diff,
            "n": n,
            "n_boot": n_boot,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": mean_diff,
        "lo": float(lo),
        "hi": float(hi),
        "n": n,
        "n_boot": int(n_boot),
    }
