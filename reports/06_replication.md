# Experiment 6: Replication on Ouro-1.4B-Thinking

**Status:** done. **Data:** `results/Ouro-1.4B/replication.json`, `results/Ouro-1.4B/replication_examples.jsonl`. **Log:** `logs/20261025-023034_replicate.log`.

[Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) is Ouro-1.4B after supervised fine-tuning on about 8.3M reasoning examples (model card). We asked whether the *same neuron indices* still steer the same concepts after post-training. There was no re-selection and no new atlas.

## Design
- **Protocol:** identical to the confirmatory test (100 held-out prompts × 3 paired samples, same α, same concept directions from the *base* atlas, same clamp values) with a fresh seed base (+2000).
- **Smaller null:** the random-neuron null uses R = 20 instead of 100, to keep the run near 30 minutes. With R = 20 the smallest attainable permutation p is 1/21 = 0.048, which cannot pass Holm at K = 5 (0.048 × 5 = 0.24). So the automated `significant` flag in `replication.json` is false for every neuron *by construction*.
- **Replication criterion (set before running):**
  - Holm-adjusted Wilcoxon p < 0.05;
  - the neuron's 50-prompt statistic exceeds all 20 random same-layer neurons;
  - PPL ratio ≤ 2;
  - distinct-2 ≥ 0.6.

## Neuron correspondence
Cosine between base and Thinking weights of each neuron (gate row, up row, down column), read from the safetensors files:

| layer | 1st / 50th / 99th percentile over the layer's 5,632 neurons (minimum of the three cosines) | candidate (gate / up / down) |
|---|---|---|
| 18 | 0.994 / 0.996 / 0.998 | N3066: 0.998 / 0.997 / 0.997; N1735: 0.998 / 0.997 / 0.998 |
| 13 | 0.993 / 0.995 / 0.998 | N1680: 0.996 / 0.998 / 0.998 |
| 4 | 0.995 / 0.997 / 0.998 | N2104: 0.996 / 0.998 / 0.998 |
| 6 | 0.995 / 0.996 / 0.998 | N3344: 0.996 / 0.997 / 0.997 |

Max activation on the base atlas's top contexts, base → Thinking:
- L18 N3066: 4.06 → 4.13
- L13 N1680: 0.66 → 0.69
- L4 N2104: 0.57 → 0.57
- L6 N3344: 0.27 → 0.25
- L18 N1735: 3.63 → 3.52

Post-training barely moves MLP neurons: identities and activation scales are preserved.

## Results
| neuron | base mean ΔC | **Thinking mean ΔC [95% CI]** | d_z | Holm p_Wilcoxon | beats random neurons | null mean / max | PPL ratio | distinct-2 | MiniLM ΔC | replicates |
|---|---|---|---|---|---|---|---|---|---|---|
| L18 N3066 − | 0.179 | 0.171 [0.146, 0.195] | 1.38 | 5.2e-17 | 20/20 | 0.004 / 0.017 | 1.79 | 0.88 | 0.244 | yes |
| L13 N1680 + | 0.141 | 0.168 [0.148, 0.189] | 1.56 | 1.6e-17 | 20/20 | −0.001 / 0.067 | 1.64 | 0.89 | 0.211 | yes |
| L4 N2104 + | 0.130 | 0.161 [0.145, 0.178] | 1.88 | 1.1e-17 | 20/20 | 0.007 / 0.048 | 1.97 | 0.90 | 0.217 | yes |
| L6 N3344 − | 0.167 | 0.180 [0.160, 0.201] | 1.68 | 1.6e-17 | 20/20 | 0.020 / 0.125 | 1.79 | 0.79 | 0.133 | yes |
| L18 N1735 − | 0.127 | 0.169 [0.142, 0.196] | 1.21 | 2.1e-16 | 20/20 | 0.003 / 0.017 | 1.98 | 0.87 | 0.233 | yes |

**All five neurons replicate.**
- Effects in the post-trained model are similar or larger: ΔC 0.161–0.180 vs 0.127–0.179. They are larger for four neurons; L18 N3066 is 0.171 vs 0.179, within the CIs.
- Every neuron exceeds all 20 random same-layer neurons and passes both guards.
- Perplexity ratios are closer to the guard (1.64–1.98) than in the base model (1.58–1.83).
- The base-model clamp values were reused. They are within ±5.1% of the Thinking model's own maxima, so the intervention strength is comparable.

## Interpretation and limits
- **Supported (systematic, for these 5 neurons):** concept-steering neurons found in the base looped model keep both their weights and their steering function through reasoning SFT. This makes them stable units of analysis across the base and post-trained checkpoints.
- **Prompts.** The same confirmatory prompts were reused, with fresh sampling seeds, on a different model. The replication is therefore across checkpoints, not across prompt sets.
- **Chat formatting.** The Thinking model was prompted with raw text stems (no chat template), matching the base protocol. Its behavior in chat format was not tested.

## Learnings
- **Plan R and K together** (as in Experiment 2). A cheaper replication null has to be framed descriptively, or it will "fail" a test it cannot pass.
- **Keep output paths per run.** The replication phase initially overwrote the base model's `confirm_examples.jsonl`. The committed file was restored from git, and the code now writes `replication_examples.jsonl`.
