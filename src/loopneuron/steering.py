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
             max_rows: int = 50) -> np.ndarray:
    """Continuations [P, n_tokens] for all prompts under one condition (row chunks get chunk-specific seeds)."""
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


def generate_samples(model, prompt_ids: torch.Tensor, cond: Cond | None, T: int, n_tokens: int, seed: int,
                     k: int) -> np.ndarray:
    """k paired samples per prompt -> [P, k, n_tokens] (sample j uses seed+j in every condition)."""
    return np.stack([generate(model, prompt_ids, cond, T, n_tokens, seed + j) for j in range(k)], axis=1)


def random_same_layer(atlas: dict, layer: int, exclude: int, n: int, seed: int) -> list[int]:
    """n distinct alive neurons from `layer` (excluding the candidate), for the random-neuron null."""
    alive = np.where(~atlas["dead"][layer].numpy())[0]
    alive = alive[alive != exclude]
    return np.random.default_rng(seed).choice(alive, size=n, replace=False).tolist()
