"""Read and steer individual SwiGLU MLP neurons of a looped transformer, per loop.

Local: one `register_forward_pre_hook` per `layers[l].mlp.down_proj`. A pre-hook sees the tensor entering
`down_proj`, i.e. a = silu(gate_proj x) * up_proj x  (the neuron activations, [batch, seq, d_mlp]), and may
return a replacement (PyTorch docs: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook).
Ouro runs the same layers T times per forward pass, so the k-th call of layer l's hook is loop k mod T.

Global context: the atlas (atlas.py) passes a `recorder`; steering experiments (steering.py) pass `edits`.
Neuron = (layer, index) as in Goodfire's "Neural Networks Are Already Clean" (https://already-clean.goodfire.com/).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

Recorder = Callable[[int, int, torch.Tensor], None]  # (layer, loop, activations [B,S,d_mlp]) -> None


@dataclass
class Edit:
    """Edit neurons[j] in batch rows[j] to values[j] (clamp), multiply (scale) or add (add).

    rows=None edits every batch row; loops=None edits at every loop (0-based loop indices otherwise).
    Vector form lets one batch carry different steered neurons per row (used for random-neuron nulls).
    """

    neurons: torch.Tensor  # [n] neuron indices within the layer
    values: torch.Tensor  # [n] clamp target / scale factor / additive offset
    mode: str = "clamp"  # "clamp" | "scale" | "add"
    rows: torch.Tensor | None = None  # [n] batch rows, or None = all rows
    loops: tuple[int, ...] | None = None  # loops at which the edit is active

    @classmethod
    def single(cls, neuron: int, value: float, mode: str = "clamp", loops: tuple[int, ...] | None = None) -> "Edit":
        """Convenience: one neuron, all batch rows."""
        return cls(neurons=torch.tensor([neuron]), values=torch.tensor([float(value)]), mode=mode, loops=loops)


def apply_edits(act: torch.Tensor, edits: list[Edit], loop: int) -> torch.Tensor:
    """Return a copy of `act` [B,S,d] with all edits active at `loop` applied (input is never modified in place)."""
    active = [e for e in edits if e.loops is None or loop in e.loops]  # loop-restricted steering (H2)
    if not active:
        return act  # untouched tensor -> bit-identical forward pass
    out = act.clone()  # never mutate the module's input tensor
    for e in active:
        n = e.neurons.to(act.device)
        v = e.values.to(device=act.device, dtype=act.dtype)
        if e.rows is None:  # same edit for every row: index the last dim only -> [B,S,n]
            cur = out[..., n]
            out[..., n] = v.expand_as(cur) if e.mode == "clamp" else (cur * v if e.mode == "scale" else cur + v)
        else:  # paired advanced indexing: row r_j, neuron n_j -> [n,S]
            r = e.rows.to(act.device)
            cur = out[r, :, n]
            v2 = v[:, None]  # broadcast over sequence positions
            out[r, :, n] = v2.expand_as(cur) if e.mode == "clamp" else (cur * v2 if e.mode == "scale" else cur + v2)
    return out


class NeuronHooks:
    """Context manager attaching record/steer pre-hooks to every layer's mlp.down_proj.

    Usage: `with NeuronHooks(model, T, recorder=fn, edits={layer: [Edit, ...]}): model(ids)`.
    The per-layer call counter is reset on enter, so the loop index is exact for any number of forward calls
    (including cached generation: every forward pass calls each down_proj exactly T times).
    """

    def __init__(self, model, ut_steps: int, recorder: Recorder | None = None, edits: dict[int, list[Edit]] | None = None):
        self.layers = [layer.mlp.down_proj for layer in model.model.layers]  # Ouro: model.model.layers[l].mlp
        self.T = ut_steps
        self.recorder = recorder
        self.edits = edits or {}
        self.calls = [0] * len(self.layers)  # how many times each layer's down_proj ran
        self._handles: list = []

    def _make_hook(self, layer: int):
        def hook(module, args):
            loop = self.calls[layer] % self.T  # k-th call of this layer -> loop k mod T
            self.calls[layer] += 1
            act = args[0]  # [B,S,d_mlp] activations entering down_proj
            if layer in self.edits:
                act = apply_edits(act, self.edits[layer], loop)
            if self.recorder is not None:
                self.recorder(layer, loop, act)  # record what down_proj actually receives (post-edit)
            return (act,) + tuple(args[1:])  # replacement positional args for down_proj

        return hook

    def __enter__(self) -> "NeuronHooks":
        self.calls = [0] * len(self.layers)
        self._handles = [m.register_forward_pre_hook(self._make_hook(i)) for i, m in enumerate(self.layers)]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()  # always detach, even on exceptions -> no stale steering
        self._handles = []
