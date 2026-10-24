"""Statistics helpers must have correct size, power direction, effect sizes and CI coverage (synthetic data)."""
import numpy as np

from loopneuron import stats


def test_wilcoxon_detects_shift_and_controls_size():
    rng = np.random.default_rng(0)
    assert stats.wilcoxon_greater(rng.normal(0.5, 1, 100)) < 1e-3
    null_p = np.array([stats.wilcoxon_greater(rng.normal(0, 1, 60)) for _ in range(400)])
    assert 0.02 < (null_p < 0.05).mean() < 0.09  # nominal size 0.05
    assert stats.wilcoxon_greater(rng.normal(-0.5, 1, 100)) > 0.99  # one-sided


def test_perm_p_formula():
    null = np.arange(99, dtype=float)  # 0..98
    assert stats.perm_p(100.0, null) == 1 / 100  # nothing >= observed
    assert stats.perm_p(49.5, null) == (1 + 49) / 100


def test_holm_matches_manual():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    # manual Holm: sorted 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08->max(0.09), 0.2*1=0.2
    assert np.allclose(stats.holm(p), [0.04, 0.09, 0.09, 0.2])


def test_effect_sizes():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert stats.rank_biserial(x) == 1.0
    assert stats.rank_biserial(-x) == -1.0
    assert np.isclose(stats.cohen_dz(x), x.mean() / x.std(ddof=1))


def test_bootstrap_ci_coverage():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(150):
        lo, hi = stats.bootstrap_ci(rng.normal(0, 1, 50), np.mean, n_boot=1000, seed=int(rng.integers(1e9)))
        hits += lo <= 0 <= hi
    assert 0.88 < hits / 150 <= 1.0


def test_required_n_monotone():
    n_small = stats.required_n(0.8, alpha=0.05, power=0.9)
    n_big = stats.required_n(0.3, alpha=0.05, power=0.9)
    assert 10 < n_small < n_big
    assert stats.required_n(0.5, alpha=0.05 / 10, power=0.9) > stats.required_n(0.5, alpha=0.05, power=0.9)
