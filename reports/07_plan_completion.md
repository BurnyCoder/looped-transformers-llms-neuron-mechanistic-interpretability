# Experiment 7: Plan-completion analyses (secondary; run after the confirmatory decision)

**Status:** done. **Data:** `results/Ouro-1.4B/extras.json`. **Log:** `logs/*_extras.log`.

A re-check of the approved plan found several items that were planned but missing from the main pass. They were run here on the 5 confirmed neurons, at their pre-registered α unless stated. None of them feeds back into the confirmatory decision.

## 1. Seed sensitivity: 3 new, independent seed sets
Each set uses 100 confirm prompts, k = 1 sample, and seeds 5234, 6234 and 7234, none of them used before.

| neuron | ΔC, seed set 1 | ΔC, seed set 2 | ΔC, seed set 3 | largest p (Wilcoxon) |
|---|---|---|---|---|
| L18 N3066 − | 0.141 | 0.165 | 0.155 | 1.7e-11 |
| L13 N1680 + | 0.151 | 0.132 | 0.153 | 9.1e-14 |
| L4 N2104 + | 0.129 | 0.136 | 0.139 | 7.5e-15 |
| L6 N3344 − | 0.181 | 0.160 | 0.158 | 9.1e-16 |
| L18 N1735 − | 0.119 | 0.113 | 0.113 | 2.9e-08 |

The effect is stable across sampling seeds. The spread between seed sets is 0.006–0.024 ΔC, against effects of 0.11–0.18. All 15 tests give p ≤ 2.9e-8. Bootstrap CIs are in `extras.json`.

## 2. Dose α = 0 (neuron held at 0 during generation; 60 prompts)
ΔC = −0.001, +0.001, +0.002, −0.001 and −0.005 for the five neurons, with all p ≥ 0.38.

Silencing a neuron does not reduce the concept score on these neutral prompts. This is expected, since the unsteered continuations carry almost none of the concept. The steering effect comes from turning the neuron *up*, and the opposite-sign clamps in [Experiment 4](04_followups.md) agree.

## 3. Concept-token probability shift (teacher-forced)
**Keywords** are the 20 tokens most over-represented within ±3 tokens of the neuron's top-40 atlas peaks, relative to 5,000 random windows (smoothed log-odds).

**Measurement.** On prompt + unsteered continuation (100 prompts), we compare the mean log of the total next-token probability on those keywords with the neuron clamped vs not.

| neuron | top keywords (decoded) | Δ log p(keywords) (nats) [95% CI] | p |
|---|---|---|---|
| L18 N3066 − | sounds, vowel, vowels, ow, word, V, sound | +0.83 [0.76, 0.90] | 1.9e-18 |
| L13 N1680 + | nothing, being, but, which, one, is, it | +0.45 [0.40, 0.49] | 2.0e-18 |
| L4 N2104 + | pedagogy, pedagogical, Pedagog, agog, Content, instruction, knowledge | +0.49 [0.44, 0.54] | 2.0e-18 |
| L6 N3344 − | ikes, Like, ADD, Enter, answers, answer | +0.94 [0.84, 1.04] | 1.9e-18 |
| L18 N1735 − | orbit, outermost, shell, orbital, s, d | +0.83 [0.75, 0.91] | 1.9e-18 |

- Clamping one neuron multiplies the probability mass on its concept vocabulary by e^0.45 to e^0.94, i.e. 1.6–2.6×, at every position.
- This is a direct, LLM-free and embedding-free confirmation of the steering effect at the level of next-token predictions.
- L13 N1680's keywords are mostly function words ("nothing, being, which, one"). Its top contexts are abstract philosophical prose whose distinctive words are spread out, not concentrated near the peak, so its token-level signal is the least specific.

## 4. Self-repair ratio (H2, mechanistic)
For a clamp at a single loop t, on 60 prompts, teacher-forced:
- **injected** = relative residual displacement right after the steered layer at loop t;
- **surviving** = relative displacement of the final normalized hidden state;
- **ratio** = surviving / injected.

| neuron | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---|---|---|
| L18 N3066 − | 0.51 | 0.42 | 0.42 | 1.26 |
| L13 N1680 + | 0.78 | 0.61 | 0.58 | 1.14 |
| L4 N2104 + | 0.46 | 0.42 | 0.49 | 0.71 |
| L6 N3344 − | 0.41 | 0.36 | 0.39 | 0.67 |
| L18 N1735 − | 0.51 | 0.43 | 0.45 | 1.19 |
| L19 N3901 − (exploratory) | 0.40 | 0.33 | 0.41 | 1.17 |

- An injection at loops 1–3 shrinks to 33–78% of its size by the time it reaches the output, with the minimum usually at loop 2. An injection at the last loop is kept or amplified (67–126%).
- This quantifies the self-repair inferred from behavior in [Experiment 4](04_followups.md): later loops actively shrink the perturbation left by an earlier loop.
- It still leaves a 22–67% residue. The residue is not enough to steer behavior from a single loop, but repeated injection at every loop accumulates.

## 5. Saturation fit of the T sweep (H3)
We fit y(T) = a − b·exp(−T/τ) to the T-sweep data (T = 1…8) with `scipy.optimize.curve_fit`.

| neuron | next-token KL: τ (loops), R² | ΔC: R² |
|---|---|---|
| L18 N3066 − | 2.0, 0.99 | 0.46 |
| L13 N1680 + | 3.9, 0.98 | 0.86 |
| L4 N2104 + | 2.0, 0.94 | 0.83 |
| L6 N3344 − | 2.1, 0.90 | 0.30 (degenerate fit) |
| L18 N1735 − | 2.5, 0.99 | 0.34 (degenerate fit) |
| L19 N3901 − (expl.) | 0.5, 0.97 | 0.31 |

- The **mechanistic effect** (KL) saturates with the number of loops, with a time constant of about 2–4 loops. Clamping at more loops helps, but with diminishing returns.
- The **behavioral effect** (ΔC) is already near its plateau at T = 1–2, so a saturating curve explains little variance.

## 6. Interpretability audit: Wilson 95% CIs (H4)
Share of neurons whose same-loop explanation reaches detection balanced accuracy ≥ 0.75, among parsable scores:
- **loop 1:** 32/128 = 25% [18%, 33%];
- **loop 4:** 28/135 = 21% [15%, 28%].

The earlier reports said 21% and 19% because they counted unparsable neurons as failures; those reports have been corrected. Goodfire's ≈ 90% "one main job" figure for Qwen3-32B comes from a different procedure, a stronger explainer plus human audits, and is not comparable.

## 7. Positional-neuron check (planned screening filter, done post hoc)
For all 10 validated candidates:
- top-40 peaks span positions 4–255 of the 256-token chunks;
- 35–38 distinct positions are used;
- no single position holds more than 7% of the peaks.

None is a positional neuron, so the missing filter did not affect the selection.

## Learnings
- **Re-checking the plan at the end caught real gaps.** One gap also exposed a reporting error: unparsable LLM scores had been counted as failures.
- **Teacher-forced token-probability shifts are the cheapest, most direct steering metric** for single neurons, and need neither an embedder nor a judge.
