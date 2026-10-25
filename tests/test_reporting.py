"""Audit and LLM-judge statistics in reporting must pair the right scores, drop unparsable ones and match scipy/Holm."""
import numpy as np
from scipy import stats as st

from loopneuron import reporting, stats


def _rec(s0, s3, c0, c3, sh):
    """One audit.json record (JSON keys are strings; None/NaN = unparsable detection score)."""
    ba = lambda v: {"balanced_acc": v}
    return {"same_loop": {"0": ba(s0), "3": ba(s3)}, "cross_loop": {"0": ba(c0), "3": ba(c3)}, "shuffled": ba(sh)}


def test_audit_tests_pairing_and_nan_handling():
    audit = [_rec(0.9, 0.8, 0.7, 0.75, 0.5), _rec(0.6, float("nan"), 0.5, 0.6, 0.4), _rec(0.7, 0.9, 0.8, 0.6, None),
             _rec(0.8, 0.7, 0.6, 0.65, 0.55), _rec(0.65, 0.85, 0.6, 0.7, 0.45)]
    out = reporting.audit_tests(audit, T=4)
    # same - cross for the loop-1 and loop-4 explanations of each record; the NaN loop-4 score of record 2 is dropped
    sc = np.array([0.9 - 0.7, 0.8 - 0.75, 0.6 - 0.5, 0.7 - 0.8, 0.9 - 0.6, 0.8 - 0.6, 0.7 - 0.65, 0.65 - 0.6, 0.85 - 0.7])
    assert out["same_minus_cross"]["n"] == 9 and np.isclose(out["same_minus_cross"]["mean"], sc.mean())
    assert np.isclose(out["same_minus_cross"]["p_two_sided"], st.wilcoxon(sc).pvalue)
    ss = np.array([0.8 - 0.5, 0.7 - 0.55, 0.85 - 0.45])  # loop-4 same - shuffled (shuffled is scored on loop-4 contexts)
    assert out["same_last_minus_shuffled"]["n"] == 3 and np.isclose(out["same_last_minus_shuffled"]["mean"], ss.mean())


def test_judge_holm_matches_stats_holm():
    judge = {"a": {"p_concept": 0.003}, "b": {"p_concept": 0.04}, "c": {"p_concept": 0.0001}}
    assert np.allclose(list(reporting.judge_holm(judge).values()), stats.holm([0.003, 0.04, 0.0001]))
