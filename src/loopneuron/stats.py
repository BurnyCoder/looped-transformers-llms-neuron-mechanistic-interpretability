"""Statistical tests for the pre-registered steering analysis.

Local: thin, tested wrappers around SciPy/statsmodels so every number in reports comes from one place.
- one-sided (steering) and two-sided (audit) Wilcoxon signed-rank (scipy.stats.wilcoxon: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
- empirical permutation p-value with the +1 correction (Phipson & Smyth 2010, https://arxiv.org/abs/1603.05766)
- Holm step-down correction (statsmodels multipletests: https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)
- matched-pairs rank-biserial correlation (Kerby 2014, doi:10.2466/11.IT.3.1) and Cohen's d_z
- percentile bootstrap CI (Efron & Tibshirani 1993)
- power analysis for a one-sided paired t-test (statsmodels TTestPower), inflated by 1/ARE for Wilcoxon (ARE=3/pi~0.955)
Global context: steering.py produces per-prompt deltas; this module turns them into the decision table.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

WILCOXON_ARE = 3 / math.pi  # asymptotic relative efficiency of Wilcoxon vs t under normality (~0.955)


def wilcoxon_greater(x: np.ndarray) -> float:
    """One-sided p-value for H1: median of paired differences x > 0 (zeros dropped, 'wilcox' convention)."""
    x = np.asarray(x, dtype=float)
    if np.all(x == 0):
        return 1.0  # no information
    return float(st.wilcoxon(x, alternative="greater", zero_method="wilcox").pvalue)


def wilcoxon_two_sided(x: np.ndarray) -> float:
    """Two-sided p-value for H1: median of paired differences x != 0 (non-directional comparisons, e.g. the H4 audit)."""
    x = np.asarray(x, dtype=float)
    return 1.0 if np.all(x == 0) else float(st.wilcoxon(x, zero_method="wilcox").pvalue)


def perm_p(observed: float, null: np.ndarray) -> float:
    """Empirical one-sided p = (1 + #{null >= observed}) / (R + 1); never 0 (Phipson & Smyth 2010)."""
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def holm(pvals) -> np.ndarray:
    """Holm-adjusted p-values (family-wise error control across candidate neurons)."""
    return multipletests(np.asarray(pvals, dtype=float), method="holm")[1]


def cohen_dz(x: np.ndarray) -> float:
    """Paired standardized effect: mean(x) / sd(x) with ddof=1."""
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else float("inf") * np.sign(x.mean())


def rank_biserial(x: np.ndarray) -> float:
    """Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-) over non-zero differences (Kerby 2014)."""
    x = np.asarray(x, dtype=float)
    x = x[x != 0]
    if len(x) == 0:
        return 0.0
    r = st.rankdata(np.abs(x))
    w_pos, w_neg = r[x > 0].sum(), r[x < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap (1-alpha) CI of `stat` over resampled prompts."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def required_n(dz: float, alpha: float, power: float = 0.9) -> int:
    """Prompts needed for a one-sided paired test with effect d_z at level alpha (Wilcoxon-adjusted)."""
    n_t = TTestPower().solve_power(effect_size=abs(dz), alpha=alpha, power=power, alternative="larger")
    return int(math.ceil(n_t / WILCOXON_ARE))


def summarize(delta: np.ndarray, null_means: np.ndarray, seed: int = 0) -> dict:
    """All per-candidate statistics for one steering condition (before multiplicity correction)."""
    delta = np.asarray(delta, dtype=float)
    lo, hi = bootstrap_ci(delta, seed=seed)
    return {
        "n": int(len(delta)), "mean_delta": float(delta.mean()), "ci95": [lo, hi],
        "dz": cohen_dz(delta), "rank_biserial": rank_biserial(delta),
        "p_wilcoxon": wilcoxon_greater(delta),
        "p_null": perm_p(float(delta.mean()), null_means), "null_mean": float(np.mean(null_means)),
        "null_q95": float(np.quantile(null_means, 0.95)),
    }
