# Methodology (full detail)

The README gives a short summary. This file holds the technical detail behind each choice, with sources. Pre-registered quantities, such as candidate neurons, α, sample sizes and decision rules, are frozen in [`reports/preregistration.md`](../reports/preregistration.md) before the confirmatory run.

## 1. Object of study: Ouro-1.4B

These facts come from `config.json` and `modeling_ouro.py` at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` of [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B):

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

**Determinism.** `OuroForCausalLM.forward` uses `early_exit_threshold = 1.0` by default. It computes a cumulative sum over a per-loop exit distribution held in the model dtype (bf16), so rounding can make the threshold trip before the last loop. All our runs pass `exit_at_step = T − 1`. This makes the computation identical for every token, with exactly T loops, and a unit test checks it.

## 2. Neuron definition

- The neuron (ℓ, i) at loop t on token position p is `a[ℓ,i,t,p] = silu(g_i·x) · (u_i·x)`, where x = `post_attention_layernorm(h)`.
- This is the tensor that enters `down_proj`. It is read with a `register_forward_pre_hook` on `layers[ℓ].mlp.down_proj`. Since `down_proj` is called once per loop per layer, the k-th call within a forward pass is loop `k mod T`.
- Activations are signed. Following Goodfire's practice for SwiGLU models, we study the **positive pole** (largest values) and the **negative pole** (most negative values) separately.

The neuron's write to the residual stream is `post_attention_layernorm_2(W_down a)`. This write is RMS-normalized *after* the MLP.

**Prediction (sandwich-norm saturation).** Clamp a_i to c, so that W_down a = c·w_i + r, where r is the contribution of the other neurons. As |c| → ∞, the write tends to `γ ⊙ sign(c)·w_i / rms(w_i)`, where γ is the norm gain. Two consequences follow:
1. The steering strength per layer per loop is bounded, so the dose-response must plateau. The crossover scale is c\* ≈ ‖r‖/‖w_i‖.
2. A very large clamp also suppresses the rest of that layer's MLP output (a "takeover" effect).

Random same-layer neurons clamped at the same relative α undergo the same takeover. That makes them the right null for separating *concept-specific* effects from *generic MLP-takeover* effects.

## 3. Data
- **Atlas corpus.** FineWeb-Edu (`sample-10BT`) and OpenWebMath, streamed with `datasets` (`streaming=True`), are tokenized with Ouro's tokenizer and packed into 256-token chunks with a BOS token. Ouro's pre-training mix includes web, code and math ([model card](https://huggingface.co/ByteDance/Ouro-1.4B)), so this text is in distribution. We use 8M tokens. Goodfire used 100M tokens for Qwen3-32B; our smaller budget is a stated limitation.
- **Prompt corpus.** WikiText-103 *test* paragraphs ([Merity et al. 2016](https://arxiv.org/abs/1609.07843)) are truncated to exactly 12 tokens. Equal lengths let us batch without padding. The prompts are split once with a fixed seed into discovery and confirmatory sets, which are disjoint.

## 4. Atlas statistics
For every neuron the atlas stores:
- **Top-k contexts per pole.** k = 40 overall, using the maximum over positions × loops and recording the loop of the maximum. Separately, k = 20 per loop.
- **Per-loop summaries.** The mean, the mean of |a|, and the firing frequency, defined as the fraction of tokens with |a| above 10% of the neuron's running maximum.
- **Histogram.** A log-binned histogram of |a|.
- **Dead flag.** A neuron is flagged dead if its max |a| stays below 1e-3.

Streaming top-k works per batch. Each sequence is reduced to its maximum per neuron, then merged with the running top-k by `torch.topk` over the concatenation. Unit tests check this against brute force.

## 5. Candidate screen (discovery data only)
1. **Coherence.** Embed each neuron's top contexts, a ±16-token window around the peak, with bge-small-en-v1.5. Coherence is the mean pairwise cosine. Neurons whose top contexts are dominated by a single token, measured by the entropy of the peak-token distribution, or by a single position are excluded as token or position detectors.
2. **Quick steer.** Take a random pool of 4,000 alive neurons (both poles, so 8,000 units). Clamp each of the 200 most coherent units at α ∈ {1, 2, 4} on 25 discovery prompts (1 sample, 32 tokens), three conditions per `generate` call. Each condition is paired with the same row block of a tiled unsteered batch. Measure ΔC (section 6) and ΔNLL, and keep the best α that passes the fluency guard (median PPL ratio ≤ 2).
3. Keep the top K ≤ 5 units by ΔC. K is capped so that the permutation test can pass Holm at all: with R = 100 random neurons the smallest attainable p is 1/101 ≈ 0.0099, and Holm requires 0.0099 ≤ 0.05/K, which holds only for K ≤ 5. Selecting on discovery data is allowed, because the confirmatory test uses only fresh prompts and seeds.
4. **Validation.** Re-run each candidate at its chosen α on 100 *other* discovery prompts (48 tokens). This gives effect sizes that are not inflated by selection (the winner's curse), and they feed the power analysis. Candidates with validation mean ΔC ≤ 0 or median ΔNLL > log 2 are dropped before pre-registration.

## 6. Metrics
- **Concept score C.** For each neuron, we build a contrastive concept direction from bge-small-en-v1.5 embeddings: u = normalize(mean emb(top contexts) − mean emb(random atlas contexts)). Then C(text) = cos(emb(text), u). The primary statistic is ΔC = C(steered continuation) − C(unsteered continuation), averaged over the k = 3 paired samples of each prompt.
- **Fluency guard.** We compute the perplexity of the steered continuation under the *unsteered* model. Within each prompt the score is the ratio steered/unsteered, and across prompts we take the median.
- **Secondary metrics:**
  - An AxBench-style blind LLM judge ([Wu et al. 2025](https://arxiv.org/abs/2501.17148)) scores concept and fluency from 0 to 2, writing its explanation before the score. The order of conditions is shuffled.
  - The shift in summed probability of concept tokens, defined as the most frequent peak tokens of the neuron's top contexts.
  - The distinct-2 repetition rate.
  - ΔC computed with all-MiniLM-L6-v2, as a robustness check on the embedder.

## 7. Statistics
- **Paired test.** A one-sided Wilcoxon signed-rank test on per-prompt ΔC, steered vs unsteered (`scipy.stats.wilcoxon`, `alternative="greater"`).
- **Random-neuron null.** Draw R = 100 alive neurons from the candidate's layer. Steer each at the same α relative to its own max |a|, on the same pole sign convention. Score each with the *candidate's* concept direction. The empirical p-value is (1 + #{null ≥ observed}) / (R + 1). This follows Goodfire's random same-layer control.
- **Multiple comparisons.** Holm correction across the K candidates (`statsmodels.stats.multitest.multipletests`).
- **Effect sizes.** d_z (mean ΔC / sd ΔC) and the matched-pairs rank-biserial correlation, each with a 10,000-sample bootstrap 95% CI.
- **Power.** `statsmodels.stats.power.TTestPower` is run on the discovery d_z with α = 0.05/K and power 0.9. The Wilcoxon test's asymptotic relative efficiency vs the t-test is ≈ 0.955 under normality, so the required n is inflated by 1/0.955.
- **Seed sensitivity.** The confirmatory test is repeated with 3 independent seed sets, and each is reported.

## 8. Loop dynamics
- **H2 (loop locus).** Clamp only at loop t ∈ {1, 2, 3, 4}. Prior residual-direction steering of Ouro-1.4B-Thinking found the last loop most effective, and found that earlier injections get "rotated away" ([nesiacel 2026](https://www.lesswrong.com/posts/rzcJMpnehFBD686gj/a-recurrent-llm-is-quite-easy-to-interpret-but-complex-to)). We test whether this holds for single neurons. We also measure how much of the injected direction w_i survives at the final norm.
- **H3 (test-time loops).** Clamp at all loops while varying T ∈ {1..8}. T > 4 is extrapolation beyond training ([Zhu et al. 2025](https://arxiv.org/abs/2510.25741)).

## 9. Interpretability audit (H4)
- Sample 300 alive neurons, stratified by layer tercile.
- For each neuron and each loop, the local LLM writes an explanation from 20 top examples at that loop. It uses the Delphi explainer format, with the peak token marked ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)).
- **Detection score.** The LLM reads 10 held-out activating contexts and 10 random contexts, and decides for each one whether the explanation applies. The score is the balanced accuracy.
- **Controls:**
  - same loop vs cross loop (the loop-1 explanation scored on loop-4 examples);
  - a shuffled explanation, taken from another neuron, as a chance baseline.
- **Reporting.** Fractions come with Wilson CIs, and stages are compared with Fisher exact tests.

## 10. Replication
[Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) has the same architecture. We first check that neuron identity is preserved, using the per-neuron cosine of the gate, up and down weight vectors between the two checkpoints. We then re-run the confirmatory protocol for the significant neuron(s) on fresh prompts and seeds.
