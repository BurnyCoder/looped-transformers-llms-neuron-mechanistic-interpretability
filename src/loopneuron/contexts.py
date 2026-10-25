"""Turn atlas entries (chunk id, position) into readable text windows and Delphi-style marked examples.

Local: the atlas stores only (seq, pos, loop, value); text is recovered from the cached token chunks.
Marked examples follow the Delphi explainer convention of delimiting the activating token with << >>
(Paulo et al. 2024, https://github.com/EleutherAI/delphi).
Global context: used by screening (coherence), metrics (concept directions), autointerp (prompts) and viewer.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch


def neuron_entries(atlas: dict, layer: int, pole: int, neuron: int, loop: int | None = None, k: int | None = None):
    """List of (seq, pos, value, loop) for a neuron's top contexts (overall, or at one loop if `loop` given)."""
    if loop is None:  # loop-pooled table: also records the loop where the max occurred
        v, s, p, l = (atlas[f"top_{x}"][layer, pole, neuron] for x in ("v", "seq", "pos", "loop"))
        rows = [(int(a), int(b), float(c), int(d)) for c, a, b, d in zip(v, s, p, l)]
    else:  # per-loop table
        v, s, p = atlas["lt_v"][layer, loop, pole, neuron], atlas["lt_seq"][layer, loop, pole, neuron], atlas["lt_pos"][layer, loop, pole, neuron]
        rows = [(int(a), int(b), float(c), loop) for c, a, b in zip(v, s, p)]
    rows = [r for r in rows if np.isfinite(r[2])]  # unfilled slots stay -inf
    return rows[:k] if k else rows


def window_ids(chunks: np.ndarray, seq: int, pos: int, left: int = 24, right: int = 8) -> tuple[list[int], int]:
    """Token ids of chunk `seq` around `pos` (clipped to the chunk) and the index of `pos` inside the window."""
    lo, hi = max(0, pos - left), min(chunks.shape[1], pos + right + 1)
    return chunks[seq, lo:hi].astype(np.int64).tolist(), pos - lo


def window_text(tok, chunks, seq, pos, left=24, right=8, mark: bool = False) -> str:
    """Decoded window; with mark=True the peak token is wrapped as <<token>> (Delphi format)."""
    ids, i = window_ids(chunks, seq, pos, left, right)
    if not mark:
        return tok.decode(ids)
    return tok.decode(ids[:i]) + "<<" + tok.decode(ids[i:i + 1]) + ">>" + tok.decode(ids[i + 1:])


def random_entries(chunks: np.ndarray, n: int, seed: int, skip: int = 4) -> list[tuple[int, int]]:
    """n random (seq, pos) pairs from the atlas chunks (baseline contexts)."""
    rng = np.random.default_rng(seed)
    return list(zip(rng.integers(0, len(chunks), n).tolist(), rng.integers(skip, chunks.shape[1], n).tolist()))


def peak_token_share(chunks: np.ndarray, entries) -> float:
    """Share of top contexts whose peak token is the single most common peak token (token-detector diagnostic)."""
    toks = Counter(int(chunks[s, p]) for s, p, *_ in entries)
    return max(toks.values()) / max(1, len(entries)) if toks else 0.0


@torch.no_grad()
def token_activations(model, chunks: np.ndarray, seqs: list[int], layer: int, neuron: int, T: int) -> np.ndarray:
    """Recompute one neuron's activation at every position and loop for chunks `seqs` -> [len(seqs), T, S]."""
    from loopneuron.hooks import NeuronHooks

    out = torch.zeros(len(seqs), T, chunks.shape[1])

    def rec(l, t, a):
        if l == layer:
            out[:, t] = a[..., neuron].float().cpu()

    ids = torch.from_numpy(chunks[seqs].astype(np.int64)).cuda()
    with NeuronHooks(model, T, recorder=rec):
        model(ids, use_cache=False)
    return out.numpy()
