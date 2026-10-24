"""Steered generation: clamp one neuron (at all or selected loops) and sample paired continuations.

Local: a condition is (layer, neuron, value, loops) or None (unsteered). Each condition is generated in its own
`generate` call with an identical batch layout and seed, so steered and unsteered rows consume the same random
numbers (common-random-numbers pairing; they differ only through the intervention).
Clamp value = alpha * max|a| of the neuron's pole from the atlas (sign: + for the positive pole, - for negative),
the "turn the neuron up" intervention of Goodfire (https://already-clean.goodfire.com/) made quantitative.
Global context: used by screening, the confirmatory test, random-neuron nulls and loop-dynamics experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from loopneuron.hooks import Edit, NeuronHooks
from loopneuron.model import sample


@dataclass(frozen=True)
class Cond:
    """One steering condition (frozen -> usable as a dict key for caching)."""

    layer: int
    neuron: int
    value: float  # clamp target for the neuron's activation
    loops: tuple[int, ...] | None = None  # None = all loops

    def edits(self) -> dict[int, list[Edit]]:
        return {self.layer: [Edit.single(self.neuron, self.value, "clamp", loops=self.loops)]}


def clamp_value(atlas: dict, layer: int, pole: int, neuron: int, alpha: float) -> float:
    """alpha * (max activation on the pole), signed (pole 1 = negative pole -> negative value)."""
    m = float(atlas["top_v"][layer, pole, neuron, 0])  # stored as max of (+a) or (-a)
    return alpha * m * (1.0 if pole == 0 else -1.0)


def generate(model, prompt_ids: torch.Tensor, cond: Cond | None, T: int, n_tokens: int, seed: int,
             max_rows: int | None = None) -> np.ndarray:
    """Continuations [P, n_tokens] for all prompts under one condition (row chunks get chunk-specific seeds).

    75 rows/call: decoding is CPU-dispatch-bound (~0.3 s/step regardless of batch), and Ouro's KV cache costs
    ~0.8 MB per token per row (96 layer-loops), so 75 rows x 60 tokens (~3.5 GB) is the most that fits in 8 GB.
    The cache grows linearly with T, so rows scale as 300 // T (beyond VRAM, Windows silently spills to system RAM
    and decoding slows several-fold, as seen at T >= 5 in the dry run).
    """
    max_rows = max_rows or max(8, 300 // T)
    outs = []
    for i in range(0, len(prompt_ids), max_rows):
        ids = prompt_ids[i:i + max_rows].cuda()
        chunk_seed = seed * 1000 + i // max_rows  # same seed for the same chunk in every condition
        if cond is None:
            outs.append(sample(model, ids, n_tokens, chunk_seed).cpu())
        else:
            with NeuronHooks(model, T, edits=cond.edits()):
                outs.append(sample(model, ids, n_tokens, chunk_seed).cpu())
    return torch.cat(outs).numpy()


def generate_multi(model, prompt_ids: torch.Tensor, conds: list[Cond | None], T: int, n_tokens: int, seed: int) -> np.ndarray:
    """Several conditions in ONE generate call: prompts tiled per condition, row-specific edits -> [C, P, n_tokens].

    Pairing: block j of the batch always holds the same prompts, so a condition is paired with the unsteered run
    only if that run uses the same tiling (call with [None] * len(conds)). Used by screening for ~3x throughput.
    """
    P = len(prompt_ids)
    edits: dict[int, list[Edit]] = {}
    for j, c in enumerate(conds):
        if c is not None:
            rows = torch.arange(j * P, (j + 1) * P)
            edits.setdefault(c.layer, []).append(Edit(neurons=torch.full((P,), c.neuron), values=torch.full((P,), c.value),
                                                      rows=rows, loops=c.loops))
    ids = prompt_ids.repeat(len(conds), 1).cuda()
    with NeuronHooks(model, T, edits=edits):
        out = sample(model, ids, n_tokens, seed * 1000).cpu().numpy()
    return out.reshape(len(conds), P, n_tokens)


def generate_samples(model, prompt_ids: torch.Tensor, cond: Cond | None, T: int, n_tokens: int, seed: int,
                     k: int) -> np.ndarray:
    """k paired samples per prompt -> [P, k, n_tokens] (sample j uses seed+j in every condition)."""
    return np.stack([generate(model, prompt_ids, cond, T, n_tokens, seed + j) for j in range(k)], axis=1)


def random_same_layer(atlas: dict, layer: int, exclude: int, n: int, seed: int) -> list[int]:
    """n distinct alive neurons from `layer` (excluding the candidate), for the random-neuron null."""
    alive = np.where(~atlas["dead"][layer].numpy())[0]
    alive = alive[alive != exclude]
    return np.random.default_rng(seed).choice(alive, size=n, replace=False).tolist()
