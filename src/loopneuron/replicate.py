"""Phase: replicate confirmed neurons on Ouro-1.4B-Thinking (same architecture, post-trained from the base model).

Local:
1. neuron correspondence: cosine between base and Thinking weights of every neuron in the candidate layers
   (gate_proj row, up_proj row, down_proj column) read directly from the safetensors files (no GPU);
2. the candidate's natural max activation in the Thinking model on the base atlas's top contexts;
3. the same pre-registered confirmatory protocol (confirm.confirm) on the Thinking model with a fresh seed.
Global context: run_pipeline.py phase `replicate`; results/<model>/replication.json.
"""
from __future__ import annotations

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from loopneuron import contexts as cx


def _weights(repo: str, rev: str, layer: int) -> dict[str, torch.Tensor]:
    path = hf_hub_download(repo, "model.safetensors", revision=rev)
    with safe_open(path, framework="pt") as f:
        return {n: f.get_tensor(f"model.layers.{layer}.mlp.{n}.weight").float() for n in ("gate_proj", "up_proj", "down_proj")}


def neuron_correspondence(base: tuple[str, str], other: tuple[str, str], layer: int, neurons: list[int]) -> dict:
    """Per-neuron cosine of gate/up rows and down columns between two checkpoints (layer-wide quantiles + candidates)."""
    a, b = _weights(*base, layer), _weights(*other, layer)
    cos = {}
    for n, dim in (("gate_proj", 1), ("up_proj", 1), ("down_proj", 0)):
        cos[n] = torch.nn.functional.cosine_similarity(a[n], b[n], dim=dim)  # [5632]
    worst = torch.minimum(torch.minimum(cos["gate_proj"], cos["up_proj"]), cos["down_proj"])
    return {"layer": layer, "layer_min_cos_quantiles": np.quantile(worst.numpy(), [0.01, 0.5, 0.99]).tolist(),
            "candidates": {n: {k: float(v[n]) for k, v in cos.items()} for n in neurons}}


@torch.no_grad()
def max_activation(model, atlas, chunks, cand, T) -> float:
    """Max of the candidate's pole over its base-atlas top-40 contexts, recomputed in `model`."""
    seqs = sorted({e[0] for e in cx.neuron_entries(atlas, cand["layer"], cand["pole"], cand["neuron"])})
    acts = cx.token_activations(model, chunks, seqs, cand["layer"], cand["neuron"], T)
    return float(acts.max() if cand["pole"] == 0 else -acts.min())
