# Summary of all experiments

This file is the single overview. Details, tables and learnings are in the per-experiment reports, and every number links back to a file in `results/Ouro-1.4B/`.

## Plain-language summary
We took a 1.4-billion-parameter language model that reuses the same 24 layers **four times in a row** (Ouro-1.4B, a "looped" transformer). We looked at its individual *neurons*, the 135,168 internal units inside its feed-forward blocks.

For each neuron we found the texts that excite it most. We picked candidates that seemed to stand for a clear topic and tested whether forcing one neuron "on" makes the model write about that topic. We fixed the test plan in advance, then ran it on fresh prompts.

**Five neurons out of five passed**: forcing just one of them on pulls the model's writing toward its topic:
- letters and sounds (phonetics)
- spiritual non-duality
- teaching (pedagogy)
- "accepted answer" forum talk
- electron orbitals

Each did this far more than 100 randomly chosen neurons from the same layer, while the text stayed mostly fluent.

Forcing a neuron on at only one of the four passes barely works. The model seems to repair a one-off push, so the push has to be repeated at every pass.

## Glossary
| term | meaning here |
|---|---|
| looped / recurrent-depth model | a transformer that applies the same stack of layers several times per token ([Ouro](https://arxiv.org/abs/2510.25741): 24 layers × 4 loops) |
| loop (t) | one pass through the shared stack; Ouro was trained with T = 4 |
| neuron (ℓ, i) | element i of the SwiGLU activation `silu(gate·x)·(up·x)` entering `down_proj` in layer ℓ; it fires once per loop |
| pole | the sign of a neuron's activation; positive (+) and negative (−) poles are treated as separate units ([Goodfire](https://already-clean.goodfire.com/)) |
| atlas | the per-neuron, per-pole, per-loop top-activating text contexts collected from 8M tokens |
| clamp / steer | overwrite a neuron's activation with α × its maximum observed activation (signed by pole) at every position (and every loop unless stated) |
| concept direction u | normalize(mean embedding of the neuron's top contexts − mean embedding of random contexts), in bge-small-en-v1.5 space |
| concept score C, ΔC | cosine of a continuation's embedding with u; ΔC = steered − unsteered, paired by seed |
| random same-layer null | the same intervention applied to 100 random neurons from the candidate's layer, scored with the candidate's u |
| Holm | step-down correction controlling family-wise error across the 5 candidates |
| d_z | mean paired difference / SD of paired differences |
| distinct-2 | unique bigrams / bigrams; low values mean repetitive, degenerate text |
| PPL ratio | perplexity of the steered continuation / unsteered continuation, both scored by the unsteered model |
| super-additive | the effect of clamping all loops together exceeds the sum of single-loop effects |

## Models in this study
| model | role | architecture | params | training data | loops | neurons | here |
|---|---|---|---|---|---|---|---|
| [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) @ `574fa66` | object of study | 24-layer decoder, d = 2048, MHA-16, SwiGLU 5632, sandwich RMSNorm, early-exit gate | 1.4B | 7.7T tokens (web, code, math, long-context; model card) | 4 (shared weights) | 135,168 | atlas, selection, confirmatory test, follow-ups, audit |
| [ByteDance/Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) @ `3aaa222` | replication | same as Ouro-1.4B | 1.4B | Ouro-1.4B + SFT on ~8.3M reasoning examples (math, code, science, chat; 2 epochs; model card) | 4 | 135,168 | replication of the 5 neurons |
| [Qwen3.5-9B Q4_K_M GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) via [llama.cpp](https://github.com/ggml-org/llama.cpp) b11854 | labeler / judge | hybrid-attention decoder (Qwen3.5) | 9B (4-bit) | n/a | – | – | explanations, detection scoring, AxBench judge |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | primary embedder | BERT-small encoder | 33M | n/a | – | – | concept score |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | robustness embedder | MiniLM encoder | 23M | n/a | – | – | secondary concept score |

No model was trained or fine-tuned in this study. All experiments are inference-time interventions.

## Experiments
| # | experiment | data | key result | report |
|---|---|---|---|---|
| 1 | sanity check + neuron atlas | FineWeb-Edu + OpenWebMath, 8M tokens | loss 4.03 → 2.51 nats from T = 1 → 4; 0 dead neurons; 41% of neurons peak at loop 1, 13% at loop 4 | [01](01_setup_and_atlas.md) |
| 2 | selection + pre-registration | WikiText-103 test, 125 discovery prompts | 8,000 units → 200 coherent → 10 validated → 5 eligible; distinct-2 guard added before confirmation (repetition has low perplexity) | [02](02_selection_and_preregistration.md) |
| 3 | **confirmatory steering test (H1)** | 100 held-out prompts × 3 samples | **5/5 neurons significant**: mean ΔC 0.13–0.18, d_z 0.99–1.61, Holm p_Wilcoxon ≤ 1.1e-15, Holm p_null = 0.0495 (floor), PPL ratio ≤ 1.83 | [03](03_confirmatory_test.md) |
| 4 | follow-ups (post hoc) + LLM judge (secondary) | 60 / 50 held-out prompts | specific (own concept 1.4–3.8× the largest other); saturating dose-response; single-loop clamps ≤ 0.043 vs all-loop 0.13–0.17 (super-additive); flat in T; 4/6 necessary by ablation (exploratory); blind LLM judge concept 0 → 0.06–0.36 (all Holm p ≤ 0.042), fluency 1.50 → 0.90–1.17 | [04](04_followups.md) |
| 5 | loop structure + LLM audit (H4) | 32k held-out tokens; 150 neurons | cross-loop r(1,4) median 0.54; adjacent loops converge (0.65 → 0.84 → 0.94); H4 not supported (same-loop 0.60 vs cross-loop 0.59, p = 0.20; weak explainer) | [05](05_loop_audit.md) |
| 6 | replication on Ouro-1.4B-Thinking | 100 held-out prompts | 5/5 replicate (ΔC 0.161–0.180, Holm p ≤ 2.1e-16, each beats 20/20 random neurons); SFT leaves neuron weights nearly unchanged (cos ≥ 0.993 for 99%) | [06](06_replication.md) |

## What would have falsified H1, and what did not happen
H1 would have been rejected if no candidate had met *all* of the following after Holm correction:
- Wilcoxon p < 0.05;
- random-neuron p < 0.05;
- PPL ratio ≤ 2;
- distinct-2 ≥ 0.6.

All five met them.

The conclusion is an existence claim: such neurons exist in Ouro-1.4B and are found by our selection procedure. It does not show that a typical neuron steers.
