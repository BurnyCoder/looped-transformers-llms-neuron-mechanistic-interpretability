"""Streaming atlas top-k must equal brute force over (sequence, position, loop) (CPU, synthetic activations)."""
import torch

from loopneuron.atlas import AtlasRecorder


def test_recorder_matches_bruteforce():
    torch.manual_seed(0)
    L, T, D, S, k, kl, skip = 2, 3, 5, 9, 4, 3, 2
    batches = [torch.randn(6, T, L, S, D) for _ in range(3)]  # [B, loop, layer, S, D]
    rec = AtlasRecorder(L, D, T, k, kl, skip_pos=skip, device="cpu")
    for bi, acts in enumerate(batches):
        rec.start_batch(torch.arange(bi * 6, bi * 6 + 6))
        for t in range(T):  # Ouro order: loop-major, then layers
            for layer in range(L):
                rec(layer, t, acts[:, t, layer])
    out = rec.summary()
    allacts = torch.cat(batches)[:, :, :, skip:, :]  # [N, T, L, S', D]
    for layer in range(L):
        for pole, sgn in ((0, 1), (1, -1)):
            a = sgn * allacts[:, :, layer]  # [N, T, S', D]
            per_seq = a.amax(dim=(1, 2))  # max over loops & positions -> [N, D]
            ref_v, ref_seq = per_seq.topk(k, dim=0)
            assert torch.allclose(out["top_v"][layer, pole], ref_v.T)
            assert torch.equal(out["top_seq"][layer, pole].long(), ref_seq.T)
            # recorded loop/pos point at the actual maximum
            for d in range(D):
                for j in range(k):
                    s_, t_, p_ = (out["top_seq"][layer, pole, d, j].item(), out["top_loop"][layer, pole, d, j].item(),
                                  out["top_pos"][layer, pole, d, j].item() - skip)
                    assert torch.isclose(a[s_, t_, p_, d], ref_v[j, d])
            for t in range(T):  # per-loop tables
                lv, ls = a[:, t].amax(dim=1).topk(kl, dim=0)
                assert torch.allclose(out["lt_v"][layer, t, pole], lv.T)
                assert torch.equal(out["lt_seq"][layer, t, pole].long(), ls.T)
    assert out["n_tokens"] == 18 * (S - skip)
    assert torch.allclose(out["mean"][0, 0], allacts[:, 0, 0].reshape(-1, D).mean(0), atol=1e-5)
