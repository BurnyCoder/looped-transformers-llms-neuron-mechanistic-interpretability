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

Each did this more strongly than 99 or 100 of 100 randomly chosen neurons from the same layer, while the text stayed mostly fluent.

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
| distinct-2 | unique / total bigrams of one continuation (a per-text variant of distinct-n, [Li et al. 2016](https://arxiv.org/abs/1510.03055)); low values mean repetitive, degenerate text |
| PPL ratio | perplexity of the steered continuation / unsteered continuation, both scored by the unsteered model |
| super-additive | the effect of clamping all loops together exceeds the sum of single-loop effects |

## Models in this study
| model | role | architecture | params | training data | loops | neurons | here |
|---|---|---|---|---|---|---|---|
| [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) @ `574fa66` | object of study | 24-layer decoder, d = 2048, MHA-16, SwiGLU 5632, sandwich RMSNorm, early-exit gate | 1.4B | 7.7T tokens (web, code, math, long-context; model card) | 4 (shared weights) | 135,168 | atlas, selection, confirmatory test, follow-ups, audit |
| [ByteDance/Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) @ `3aaa222` | replication | same as Ouro-1.4B | 1.4B | Ouro-1.4B + SFT on ~8.3M reasoning examples (math, code, science, chat; 2 epochs; model card) | 4 | 135,168 | replication of the 5 neurons |
| [Qwen3.5-9B Q4_K_M GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) via [llama.cpp](https://github.com/ggml-org/llama.cpp) b11854 | labeler / judge | hybrid-attention decoder (Qwen3.5) | 9B (4-bit) | n/a | – | – | explanations, detection scoring, AxBench judge |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | primary embedder | 12-layer BERT encoder (hidden 384) | 33M | n/a | – | – | concept score |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | robustness embedder | MiniLM encoder | 23M | n/a | – | – | secondary concept score |

No model was trained or fine-tuned in this study. All experiments are inference-time interventions.

## Experiments
| # | experiment | data | key result | report |
|---|---|---|---|---|
| 1 | sanity check + neuron atlas | FineWeb-Edu + OpenWebMath, 8M tokens | loss 4.03 → 2.51 nats from T = 1 → 4; 0 dead neurons; 41% of neuron poles peak at loop 1, 13% at loop 4 | [01](01_setup_and_atlas.md) |
| 2 | selection + pre-registration | WikiText-103 test, 125 of the 150 discovery prompts | 8,000 units → 200 coherent → 10 validated → 8 eligible → top 5 by validation d_z; distinct-2 guard added before confirmation (repetition has low perplexity) | [02](02_selection_and_preregistration.md) |
| 3 | **confirmatory steering test (H1)** | 100 held-out prompts × 3 samples | **5/5 neurons significant**: mean ΔC 0.13–0.18, d_z 0.99–1.61, Holm p_Wilcoxon ≤ 1.1e-15, Holm p_null = 0.0495 (floor), PPL ratio ≤ 1.83 | [03](03_confirmatory_test.md) |
| 4 | follow-ups (post hoc) + LLM judge (secondary) | 60 / 50 held-out prompts | specific (own concept 1.4–3.7× the largest other); dose-response levels off for 4/6 neurons; single-loop clamps ≤ 0.043 vs all-loop 0.13–0.17 (super-additive); effect present at T = 1 and not growing with T; 4/6 necessary by ablation (exploratory); blind LLM judge concept 0–0.02 → 0.06–0.36 (all Holm p ≤ 0.042), fluency 1.50 → 0.90–1.17 | [04](04_followups.md) |
| 5 | loop structure + LLM audit (H4) | 32k held-out tokens; 150 neurons | cross-loop r(1,4) median 0.54; adjacent loops converge (0.65 → 0.84 → 0.94); H4 not supported (same-loop 0.60 vs cross-loop 0.59, p = 0.20; weak explainer) | [05](05_loop_audit.md) |
| 6 | replication on Ouro-1.4B-Thinking | 100 confirm prompts + 100 fresh reserve prompts | 5/5 replicate in both (confirm prompts: ΔC 0.161–0.180, Holm p ≤ 2.1e-16; fresh prompts: ΔC 0.157–0.179, Holm p ≤ 2.0e-16; each beats 20/20 random neurons); SFT leaves neuron weights nearly unchanged (cos ≥ 0.993 for 99% of neurons in the four candidate layers, ≥ 0.996 for the five neurons) | [06](06_replication.md) |
| 7 | plan-completion analyses (post hoc) | 100 / 60 held-out prompts | effect stable across 3 new seed sets (ΔC 0.11–0.18, p ≤ 2.9e-8); keyword-set probability +0.45–0.94 nats (p ≈ 2e-18); single-loop injections at loops 1–3 keep 33–78% of their size at the output vs 67–126% at loop 4 (self-repair); KL saturates in T (τ ≈ 2–4 loops); no positional candidates | [07](07_plan_completion.md) |

## Plan vs. what was executed
The approved plan is reproduced in spirit in [docs/methodology.md](../docs/methodology.md). Deviations, all decided before the confirmatory data was touched unless marked otherwise:

| plan item | executed | why |
|---|---|---|
| model: originally a <1B model (Parcae) | Ouro-1.4B + Ouro-1.4B-Thinking | user's choice |
| `transformers==4.54.1` (model card) | 4.57.6 | the pinned Ouro revision includes the KV-cache fix for ≥ 4.56 ([commit 7ea635b](https://huggingface.co/ByteDance/Ouro-1.4B/commit/7ea635ba1575ae9ab4ae1d83d83e16a6e47fe696)) and fails on 4.54.1 ("`key_cache` has no setter") |
| atlas: log-binned histograms | threshold-exceedance counts on every 8th batch | per-batch histograms were much slower in development (slowdown not logged); counts give the same coarse distribution |
| screen: drop token-identity **and positional** neurons | token detectors dropped; positional neurons checked post hoc ([Experiment 7](07_plan_completion.md)) | the positional filter was not implemented before selection |
| screen: α = 4 only, with a concept-token metric | α ∈ {1, 2, 4}; concept score only (keyword-probability shift computed post hoc for the 5 confirmed neurons) | α = 4 often broke fluency in the dry run |
| K ≤ 10 | K = 5 | with R = 100 random neurons, Holm cannot reject for K > 5 |
| decision rule: PPL guard | PPL guard **plus** distinct-2 ≥ 0.6 | repetition has low perplexity; amendment logged in the pre-registration before confirmation |
| seed sensitivity (3 seed sets), α = 0 dose, self-repair ratio, T-sweep fit, Wilson CIs | run after the main pass, in the plan-completion phase ([Experiment 7](07_plan_completion.md)) | initially missed; added when the plan was re-checked |
| audit: 300 neurons, every loop, 10 + 10 detection contexts | 150 neurons, loops 1 and 4, 8 + 8 contexts | local-LLM time on an 8 GB laptop GPU |
| LLM outputs constrained by JSON schema | published text formats parsed with regexes (Delphi `[EXPLANATION]:` and lists, AxBench `Rating: [[n]]`) | keeps the published prompt wording (system prompts and AxBench templates verbatim; Delphi few-shot examples decoded from GPT-2 byte-level BPE; no Delphi explainer few-shot demonstrations) |
| replication: same null R = 100 | R = 20 (descriptive; decided after the confirmatory test) | keeps the run near 30 minutes; the Wilcoxon test is the replication criterion (the other replication conditions were written down after the run, see [Experiment 6](06_replication.md)) |
| one PR per functional unit (7) | PRs grouped by development stage rather than one per unit: pipeline and results (#1–#4), plan audit (#5), fact-checks (#6 onward) | phases were developed and validated together in the dry run |
| CLI `--model` | the model is set in `.env` | one fewer way to diverge from the pinned revision |
| power: n = max(100, required) | n = clip(required, 100, 300) | the confirm split has 300 prompts; required n was 13–22, so n = 100 either way |
| H2 "self-repair ratio = measured effect / direct-path effect" | surviving / injected relative residual displacement for a single-loop clamp ([Experiment 7](07_plan_completion.md)), plus the behavioral super-additivity ratio ([Experiment 4](04_followups.md)) | a "direct path" is not uniquely defined in a looped model; the displacement right after the steered layer is the measurable injection |
| replication: fresh prompts and fresh seeds | (after the confirmatory test) fresh seeds on the confirm prompts (identical inputs across checkpoints), **plus** a second replication on the never-used reserve prompts ([Experiment 6](06_replication.md)) | the reserve split was free because no second confirmatory round was needed |
| test "clamping to the natural value leaves logits unchanged" | identity edits (scale = 1, add = 0) leave logits bit-identical; clamp ≡ residual add of (c − a)·w is tested separately | a per-position natural-value clamp is the same no-op, checked more simply |
| `NeuronRecorder` + `NeuronSteerer` | one `NeuronHooks` object (record and edit in the same pre-hook) | one hook per layer avoids ordering bugs between two hooks |
| viewer "both poles" | one page per candidate pole; token shading shows both signs (teal +, rust −) | candidates are poles; both signs remain visible on every page |
| disk budget 24.7 GB | the user freed ~50 GB mid-run; VRAM (8 GB) stayed the binding limit | the labeler stayed at Qwen3.5-9B Q4 because of VRAM, not disk |
| Ouro-2.6B skipped (limitation) | stated in the paper's limitations | not attempted: its bf16 checkpoint is 5.3 GB ([model files](https://huggingface.co/ByteDance/Ouro-2.6B/tree/main)) and its KV cache per token is twice Ouro-1.4B's (48 × 4 layer-loops), leaving too little of the 8 GB VRAM for the batched atlas and generation runs used here |

## What would have falsified H1, and what did not happen
H1 would have been rejected if no candidate had met *all* of the following after Holm correction:
- Wilcoxon p < 0.05;
- random-neuron p < 0.05;
- PPL ratio ≤ 2;
- distinct-2 ≥ 0.6.

All five met them.

The conclusion is an existence claim: such neurons exist in Ouro-1.4B and are found by our selection procedure. It does not show that a typical neuron steers.
