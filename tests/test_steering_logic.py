"""CPU tests for steering bookkeeping: clamp sign per pole, random-null sampling, atlas entry decoding."""

import torch

from loopneuron import contexts as cx
from loopneuron.steering import clamp_value, random_same_layer


def _toy_atlas():
    L, D, k = 2, 6, 3
    a = {"top_v": torch.tensor([[[[5.0, 4, 3]] * D, [[2.0, 1, -float("inf")]] * D]] * L),
         "top_seq": torch.zeros(L, 2, D, k, dtype=torch.int32), "top_pos": torch.full((L, 2, D, k), 4, dtype=torch.int32),
         "top_loop": torch.zeros(L, 2, D, k, dtype=torch.int32), "dead": torch.zeros(L, D, dtype=torch.bool)}
    a["dead"][1, 2] = True
    return a


def test_clamp_value_sign_follows_pole():
    a = _toy_atlas()
    assert clamp_value(a, 0, 0, 1, 2.0) == 10.0  # positive pole: +alpha*max
    assert clamp_value(a, 0, 1, 1, 2.0) == -4.0  # negative pole: -alpha*max(-a)


def test_random_same_layer_excludes_candidate_and_dead():
    a = _toy_atlas()
    picks = random_same_layer(a, 1, exclude=0, n=4, seed=0)
    assert 0 not in picks and 2 not in picks and len(set(picks)) == 4


def test_neuron_entries_drops_unfilled_slots():
    a = _toy_atlas()
    assert len(cx.neuron_entries(a, 0, 1, 0)) == 2  # third slot is -inf


