# Methodology (full detail)

The README gives a short summary. This file holds the technical detail behind each choice, with sources. Pre-registered quantities, such as candidate neurons, α, sample sizes and decision rules, are frozen in [`reports/preregistration.md`](../reports/preregistration.md) before the confirmatory run.

## 1. Object of study: Ouro-1.4B

These facts come from [`config.json`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json) and [`modeling_ouro.py`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` of [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B):

| Property | Value |
|---|---|
| layers (shared) | 24 |
| loops (`total_ut_steps`) | 4, so the effective depth is 96 layer applications |
| hidden size | 2048 |
| MLP | SwiGLU, `down_proj(silu(gate_proj x) * up_proj x)`, width 5632 |
| attention | 16-head MHA, RoPE θ = 1e6 |
| norms | sandwich RMSNorm (`input_layernorm`/`input_layernorm_2` around attention, `post_attention_layernorm`/`post_attention_layernorm_2` around the MLP), plus `model.norm` after **every** loop |
| vocab | 49,152 |
| exit | an early-exit gate gives per-loop exit probabilities; `exit_at_step` selects a fixed loop |

The loop is `for current_ut in range(total_ut_steps): for layer in layers: h = layer(h); h = norm(h)`. There is no separate input injection: the normalized output of loop *t* is the input of loop *t+1*.

**Determinism.** `OuroForCausalLM.forward` uses `early_exit_threshold = 1.0` by default. It computes a cumulative sum over a per-loop exit distribution held in the model dtype (bf16), so rounding can make the threshold trip before the last loop. All our runs set `model.early_exit_step = T − 1` (`loopneuron.model.load_model`), which the remote code uses as the default `exit_at_step`, so the logits always come from the last loop. Ouro executes all T loops either way; the exit setting only chooses which loop's hidden state feeds `lm_head`. A GPU unit test (`tests/test_hooks.py::test_default_exit_path_matches_last_loop`) checks that the default path picks the same next token at ≥ 99% of positions.

## 2. Neuron definition

- The neuron (ℓ, i) at loop t on token position p is `a[ℓ,i,t,p] = silu(g_i·x) · (u_i·x)`, where x = `post_attention_layernorm(h)`.
- This is the tensor that enters `down_proj`. It is read with a `register_forward_pre_hook` on `layers[ℓ].mlp.down_proj`. Since `down_proj` is called once per loop per layer, the k-th call within a forward pass is loop `k mod T`.
- Activations are signed. Following Goodfire's practice for SwiGLU models, we study the **positive pole** (largest values) and the **negative pole** (most negative values) separately.

The neuron's write to the residual stream is `post_attention_layernorm_2(W_down a)`. This write is RMS-normalized *after* the MLP.

**Prediction (sandwich-norm saturation).** Clamp a_i to c, so that W_down a = c·w_i + r, where r is the contribution of the other neurons. As |c| → ∞, the write tends to `γ ⊙ sign(c)·w_i / rms(w_i)`, where γ is the norm gain. Two consequences follow:
1. The steering strength per layer per loop is bounded, so the dose-response is predicted to plateau. The crossover scale is c\* ≈ ‖r‖/‖w_i‖. We measure it as the median over unsteered tokens of ‖W_down a‖/‖w_i‖ (`experiments.saturation_scale`), since the unclamped MLP output approximates r.
2. A very large clamp also suppresses the rest of that layer's MLP output (a "takeover" effect).

Random same-layer neurons clamped at the same relative α undergo the same takeover. That makes them the right null for separating *concept-specific* effects from *generic MLP-takeover* effects.

## 3. Data
- **Atlas corpus.** FineWeb-Edu ([Penedo et al. 2024](https://arxiv.org/abs/2406.17557); one parquet shard, `sample/10BT/000_00000`) and OpenWebMath ([Paster et al. 2023](https://arxiv.org/abs/2310.06786); one shard, `train-00000`), streamed with `datasets` (`streaming=True`) and interleaved per document with probabilities 0.9 / 0.1 (`interleave_datasets`; the token share was not measured), are tokenized with Ouro's tokenizer, joined with an `<|endoftext|>` separator (id 0) after each document, and packed into 256-token chunks. Held-out text for the setup check and cross-loop correlations comes from another FineWeb-Edu shard (`013_00000`). Ouro's pre-training mix includes web, code and math ([model card](https://huggingface.co/ByteDance/Ouro-1.4B)), so this text is in distribution. We use 8M tokens. Goodfire used 100M tokens for Qwen3-32B ([Cammarata 2026](https://already-clean.goodfire.com/)); our smaller budget is a stated limitation.
- **Prompt corpus.** WikiText-103 *test* paragraphs ([Merity et al. 2016](https://arxiv.org/abs/1609.07843)) are truncated to exactly 12 tokens. Equal lengths let us batch without padding. None of the 750 stems occurs verbatim in the 8M atlas tokens (exact 12-token search). The prompts are split once with a fixed seed into disjoint discovery (150), confirmatory (300) and reserve (300) sets.

## 4. Atlas statistics
For every neuron the atlas stores:
- **Top-k contexts per pole.** k = 40 overall, using the maximum over positions × loops and recording the loop of the maximum. Separately, k = 20 per loop.
- **Per-loop summaries.** The mean, the mean of |a|, the SD, and the fraction of tokens with a > 0 (`frac_pos`).
- **Exceedance counts.** For each sign, the number of tokens with |a| above each of 8 thresholds (0.01 … 30), a coarse activation distribution. Moments and exceedances are computed on every 8th batch (987,840 tokens per layer and loop); top-k uses every batch.
- **Dead flag.** A neuron is flagged dead if its max |a| stays below 1e-3.
- **Excluded positions.** The first 4 positions of every chunk are skipped (attention-sink-like outliers at the sequence start).

Streaming top-k works per batch. Each sequence is reduced to its maximum per neuron, then merged with the running top-k by `torch.topk` over the concatenation. Unit tests check this against brute force.

## 5. Candidate screen (discovery data only)
1. **Coherence.** Embed each unit's top-20 contexts, a window of 24 tokens before to 8 tokens after the peak, with [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). Coherence is the mean pairwise cosine. Units whose most common peak token covers ≥ 50% of the top-20 peaks are excluded as token detectors ([Voita et al. 2023](https://arxiv.org/abs/2309.04827)). No positional filter was applied before selection; positional neurons were checked post hoc ([Experiment 7](../reports/07_plan_completion.md)).
2. **Quick steer.** Take a random pool of 4,000 alive neurons (both poles, so 8,000 units). Clamp each of the 200 most coherent units at α ∈ {1, 2, 4} on 25 discovery prompts (1 sample, 32 tokens), three conditions per `generate` call. Each condition is paired with the same row block of a tiled unsteered batch. Measure ΔC (section 6) and ΔNLL, and keep the best α that passes the fluency guard (median PPL ratio ≤ 2).
3. Keep the top 2K = 10 units by screening ΔC for validation. K is capped so that the permutation test can pass Holm at all: with R = 100 random neurons the smallest attainable p is 1/101 ≈ 0.0099, and Holm requires 0.0099 ≤ 0.05/K, which holds only for K ≤ 5. Selecting on discovery data is allowed, because the confirmatory test uses only fresh prompts and seeds.
4. **Validation.** Re-run each candidate at its chosen α on 100 *other* discovery prompts (48 tokens). This gives effect sizes that are not inflated by selection (the winner's curse), and they feed the power analysis. Validation uses 1 sample per prompt. Candidates with validation mean ΔC ≤ 0, median ΔNLL > log 2 or median steered distinct-2 < 0.6 are dropped, and the K = 5 remaining units with the largest validation d_z are pre-registered.

## 6. Metrics
- **Concept score C.** For each neuron, we build a contrastive concept direction from bge-small-en-v1.5 embeddings: u = normalize(mean emb(top contexts) − mean emb(random atlas contexts)). Then C(text) = cos(emb(text), u). The primary statistic is ΔC = C(steered continuation) − C(unsteered continuation), averaged over the k = 3 paired samples of each prompt.
- **Fluency guard.** We compute the perplexity of the steered continuation under the *unsteered* model. Within each prompt the score is the ratio steered/unsteered, and across prompts we take the median.
- **Secondary metrics:**
  - An AxBench-style blind LLM judge ([Wu et al. 2025](https://arxiv.org/abs/2501.17148)) scores concept and fluency from 0 to 2, writing its explanation before the score. The order of conditions is shuffled.
  - distinct-2, the unique / total bigrams of each continuation (lexical diversity; a per-text variant of distinct-n from [Li et al. 2016](https://arxiv.org/abs/1510.03055), who compute it over all responses).
  - ΔC computed with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), as a robustness check on the embedder.

## 7. Statistics
- **Paired test.** A one-sided Wilcoxon signed-rank test on per-prompt ΔC, steered vs unsteered (`scipy.stats.wilcoxon`, `alternative="greater"`).
- **Random-neuron null.** Draw R = 100 alive neurons from the candidate's layer. Steer each at the same α relative to its own max |a|, on the same pole sign convention. Score each with the *candidate's* concept direction. The statistic is mean ΔC on the first 50 prompts, sample 0. The empirical p-value is (1 + #{null ≥ observed}) / (R + 1) ([Phipson & Smyth 2010](https://arxiv.org/abs/1603.05766)). This adapts Goodfire's random same-layer *ablation* control to steering. The null runs reuse the sample-0 seed but generate the 50 prompts as one batch (the unsteered baseline used 60-row batches), so their pairing with the baseline is not guaranteed bit-exact; unpaired noise can only widen the null.
- **Multiple comparisons.** Holm correction across the K candidates ([Holm 1979](https://www.jstor.org/stable/4615733); `statsmodels.stats.multitest.multipletests`).
- **Effect sizes.** Mean ΔC with a 10,000-resample percentile-bootstrap 95% CI; d_z (mean ΔC / sd ΔC) and the matched-pairs rank-biserial correlation ([Kerby 2014](https://doi.org/10.2466/11.IT.3.1)) as point estimates.
- **Power.** `statsmodels.stats.power.TTestPower` is run on the discovery d_z with α = 0.05/K and power 0.9. The Wilcoxon test's asymptotic relative efficiency vs the t-test is 3/π ≈ 0.955 under normality ([Hodges & Lehmann 1956](https://doi.org/10.1214/aoms/1177728261)), so the required n is inflated by 1/0.955.
- **Seed sensitivity (post hoc).** ΔC is re-estimated with 3 new seed sets (100 prompts, k = 1), each with a Wilcoxon p and a bootstrap CI; there is no null, Holm or guard ([Experiment 7](../reports/07_plan_completion.md)).

## 7b. Follow-ups, LLM judge and replication (as run)
- **Follow-ups (post hoc).** Run on confirm prompts 1–60, 1 sample each, with seeds distinct from the confirmatory run. They cover:
  - the specificity matrix (steer neuron i, score concept j);
  - dose-response over α ∈ {0.5, 1, 2, 4, 8, 16, −1, −4};
  - loop locus;
  - T sweep;
  - zero-ablation on atlas ranks 21–40 (19 contexts for L6 N3344) against 100 random same-layer ablations.
- **LLM judge (secondary).** Prompts 1–50. Each continuation is rated in its own call, in shuffled order, for concept (the neuron's Qwen label) and for fluency.
- **Replication.** Same protocol on Ouro-1.4B-Thinking with R = 20: on the confirm prompts with seed base +2000 (`replicate`), and on the 100 unused reserve prompts with seed base +6000 (`replicate_fresh`). With R = 20 the null is descriptive: 1/21 cannot pass Holm at K = 5, so the automated `significant` flag is false by construction. The code at the start of the first run named the Wilcoxon test vs unsteered as the replication criterion; the further conditions applied in [Experiment 6](../reports/06_replication.md) (beats all 20 random neurons, both guards pass) were written down after it and are applied manually.

- **Plan-completion analyses** (`run_pipeline.py --phase extras`, post hoc, see [Experiment 7](../reports/07_plan_completion.md)):
  - 3 new seed sets (100 prompts, k = 1);
  - α = 0 clamp;
  - teacher-forced keyword probability shift (up to 20 keywords per neuron by smoothed log-odds vs 5,000 random windows);
  - self-repair ratio (relative residual displacement right after the steered layer vs at the final norm);
  - saturating-exponential fits of the T sweep;
  - Wilson CIs for the audit;
  - a positional-neuron check of the validated candidates.

## 8. Loop dynamics
- **H2 (loop locus).** Clamp only at loop t ∈ {1, 2, 3, 4}. Prior residual-direction steering of Ouro-1.4B-Thinking found the last loop most effective, and found that earlier injections leave the residual stream perturbed but "rotated away" from the concept ([nesiacel 2026](https://www.lesswrong.com/posts/rzcJMpnehFBD686gj/a-recurrent-llm-is-quite-easy-to-interpret-but-complex-to)). We test whether this holds for single neurons. We also measure the next-token KL divergence and the relative displacement of the final hidden state that each single-loop clamp causes.
- **H3 (test-time loops).** Clamp at all loops while varying T ∈ {1..8}. T > 4 is extrapolation beyond training ([Zhu et al. 2025](https://arxiv.org/abs/2510.25741)).

## 9. Interpretability audit (H4)
- Sample 150 alive neurons, 50 per layer tercile, positive pole.
- For each neuron, at loop 1 and at loop 4, the local LLM writes an explanation from that loop's top-12 examples. It uses the Delphi explainer system prompt with the peak token marked `<<like this>>` and one 0–10 activation level per example; Delphi's few-shot demonstrations and per-token activation lists are not used ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)).
- **Detection score.** The LLM reads 8 held-out activating contexts (ranks 13–20 at the loop) and 8 random contexts, and decides for each one whether the explanation applies. The score is the balanced accuracy.
- **Controls:**
  - same loop vs cross loop (the loop-1 explanation scored on loop-4 examples);
  - a shuffled explanation, taken from another neuron, as a chance baseline.
- **Reporting.** Mean balanced accuracy with 95% CIs. Same-loop vs cross-loop (both explanations pooled) and loop-4 same-loop vs shuffled are compared with paired two-sided Wilcoxon tests (`reporting.audit_tests` → `reports/tables/audit_judge.md`). Separately, every neuron's same-token Pearson correlation between loops is computed on 32,256 held-out tokens.

## 10. Replication
[Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) has the same architecture. We first check that neuron identity is preserved, using the per-neuron cosine of the gate, up and down weight vectors between the two checkpoints. We then re-run the confirmatory protocol for the same neuron indices with fresh seeds, first on the confirm prompts (identical inputs across checkpoints) and then on the unused reserve prompts (section 7b).
