"""Tests for neuron recording/steering hooks on Ouro (written before hooks.py; TDD).

Global context: every downstream result (atlas, steering statistics) depends on the hook reading the right
tensor at the right loop. These tests pin that down against a manual SwiGLU recomputation.
"""
import pytest
import torch
import torch.nn.functional as F

from loopneuron.hooks import Edit, NeuronHooks, apply_edits

TEXT = "The mitochondria is the powerhouse of the cell, and photosynthesis happens in chloroplasts."


def test_apply_edits_clamp_equals_residual_add():
    """Clamping neuron i to c equals adding (c - a_i) * W_down[:, i] after down_proj (pure linear algebra)."""
    torch.manual_seed(0)
    down = torch.nn.Linear(8, 4, bias=False)  # stand-in for mlp.down_proj
    act = torch.randn(2, 3, 8)  # [batch, seq, neurons]
    edited = apply_edits(act, [Edit.single(neuron=5, value=3.0, mode="clamp")], loop=0)
    expected = down(act) + (3.0 - act[..., 5:6]) * down.weight[:, 5]  # residual-space identity
    assert torch.allclose(down(edited), expected, atol=1e-5)


def test_apply_edits_rows_and_loops():
    """Row-specific edits only touch their rows; loop-restricted edits only fire at their loops."""
    act = torch.zeros(3, 2, 4)
    edits = [Edit(neurons=torch.tensor([1, 2]), rows=torch.tensor([0, 2]), values=torch.tensor([5.0, -7.0]),
                  mode="clamp", loops=(1,))]
    assert torch.equal(apply_edits(act, edits, loop=0), act)  # wrong loop -> untouched
    out = apply_edits(act, edits, loop=1)
    assert out[0, :, 1].eq(5.0).all() and out[2, :, 2].eq(-7.0).all()  # targeted cells set
    assert out[1].abs().sum() == 0 and out[0, :, 2].abs().sum() == 0  # everything else untouched


def test_apply_edits_scale_and_add():
    act = torch.ones(1, 2, 3)
    assert apply_edits(act, [Edit.single(0, 0.0, "scale")], 0)[..., 0].eq(0).all()  # zero-ablation
    assert apply_edits(act, [Edit.single(2, 2.5, "add")], 0)[..., 2].eq(3.5).all()


@pytest.mark.gpu
def test_loop_counter_and_shapes(ouro):
    """down_proj is called exactly T times per layer per forward; loops are 0..T-1 in order."""
    model, tok, T = ouro
    seen = []
    ids = tok(TEXT, return_tensors="pt").input_ids.cuda()
    with NeuronHooks(model, T, recorder=lambda l, t, a: seen.append((l, t, tuple(a.shape)))), torch.no_grad():
        model(ids)
    n_layers = len(model.model.layers)
    assert len(seen) == n_layers * T
    for layer in range(n_layers):
        assert [t for l, t, _ in seen if l == layer] == list(range(T))
    assert seen[0][2] == (1, ids.shape[1], model.config.intermediate_size)


@pytest.mark.gpu
def test_recorder_matches_manual_swiglu(ouro):
    """Recorded activation == silu(gate x) * up x with x = post_attention_layernorm output, at every loop."""
    model, tok, T = ouro
    layer = 7
    mlp = model.model.layers[layer].mlp
    xs, acts = [], []
    h = model.model.layers[layer].post_attention_layernorm.register_forward_hook(lambda m, i, o: xs.append(o.detach()))
    ids = tok(TEXT, return_tensors="pt").input_ids.cuda()
    rec = lambda l, t, a: acts.append(a.detach().clone()) if l == layer else None
    with NeuronHooks(model, T, recorder=rec), torch.no_grad():
        model(ids)
    h.remove()
    assert len(xs) == len(acts) == T
    for x, a in zip(xs, acts):
        manual = F.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
        assert torch.allclose(manual.float(), a.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_identity_edits_do_not_change_logits(ouro):
    """scale=1 and add=0 edits on all layers leave logits bit-identical (hook plumbing is a no-op)."""
    model, tok, T = ouro
    ids = tok(TEXT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        base = model(ids).logits
        edits = {l: [Edit.single(3, 1.0, "scale"), Edit.single(4, 0.0, "add")] for l in range(len(model.model.layers))}
        with NeuronHooks(model, T, edits=edits):
            same = model(ids).logits
    assert torch.equal(base, same)


@pytest.mark.gpu
def test_clamp_changes_logits_and_loop_restriction(ouro):
    """A large clamp changes logits; restricting it to loop 0 vs loop T-1 gives different logits."""
    model, tok, T = ouro
    ids = tok(TEXT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        base = model(ids).logits.float()
        outs = []
        for loops in [None, (0,), (T - 1,)]:
            with NeuronHooks(model, T, edits={12: [Edit.single(100, 20.0, "clamp", loops=loops)]}):
                outs.append(model(ids).logits.float())
    assert all((o - base).abs().max() > 1e-2 for o in outs)
    assert (outs[1] - outs[2]).abs().max() > 1e-2


@pytest.mark.gpu
def test_forward_deterministic(ouro):
    model, tok, T = ouro
    ids = tok(TEXT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        assert torch.equal(model(ids).logits, model(ids).logits)


@pytest.mark.gpu
def test_kv_cache_matches_full_forward(ouro):
    """Incremental decoding with Ouro's UniversalTransformerCache reproduces full-forward next-token logits."""
    model, tok, T = ouro
    ids = tok(TEXT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        full = model(ids).logits[:, -1].float()
        pre = model(ids[:, :-1], use_cache=True)
        inc = model(ids[:, -1:], past_key_values=pre.past_key_values, use_cache=True).logits[:, -1].float()
    assert torch.allclose(full.softmax(-1), inc.softmax(-1), atol=2e-2)
    assert full.argmax(-1).item() == inc.argmax(-1).item()
