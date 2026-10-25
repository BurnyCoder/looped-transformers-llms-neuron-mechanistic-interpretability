# Experiment 2: Candidate selection on discovery data and pre-registration

**Status:** done. **Data:** `results/Ouro-1.4B/screen.json`, `results/Ouro-1.4B/screen_initial_validation.json`, `results/Ouro-1.4B/candidates.json`. **Pre-registration:** [preregistration.md](preregistration.md), committed at `5c8a44f` before any confirmatory prompt was used. **Logs:** `logs/20261024-201056_screen_candidates.log`, `logs/20261024-215718_validate_candidates.log`, `logs/20261024-220430_label_candidates.log` (full Qwen prompts and outputs), `logs/20261024-220623_preregister.log`.

## Procedure
Only discovery prompts were used: 150 stems, of which 1–25 are for screening and 26–125 for validation.

1. **Pool.** 4,000 random alive neurons × 2 poles = 8,000 units. Each unit's top-20 atlas contexts are embedded with bge-small-en-v1.5.
   - Coherence (mean pairwise cosine): median 0.498, 90th percentile 0.578.
   - 21.2% of units are *token detectors* (one token is the peak in ≥ 50% of top contexts) and are excluded ([Voita et al. 2023](https://arxiv.org/abs/2309.04827)).
2. **Quick steer.** The 200 most coherent remaining units (coherence ≥ 0.618; all 24 layers represented) are each clamped at α ∈ {1, 2, 4} × max|a| on 25 prompts, with 1 sample of 32 tokens each.
   - 197 of 200 units have at least one α whose median perplexity ratio is ≤ 2.
3. **Validation.** The top 10 units by screening ΔC are re-run at their chosen α on 100 *different* discovery prompts (48 tokens). This gives selection-free effect sizes.
4. **Eligibility and K.** A unit is eligible if mean ΔC > 0, median ΔNLL ≤ log 2, and median steered distinct-2 ≥ 0.6. The K = 5 eligible units with the largest validation d_z are kept.
   - K ≤ 5 is forced by the permutation test: the smallest attainable p with R = 100 random neurons is 1/101 = 0.0099, and that must be ≤ 0.05/K.
5. **Labels.** The local Qwen3.5-9B (llama.cpp) labels the units from their top-20 marked contexts, using the Delphi explainer prompt ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)). Labels are for documentation and the LLM judge only. The primary metric uses no LLM.
6. **Power.** One-sided paired test at α = 0.05/5, power 0.9, divided by the Wilcoxon asymptotic relative efficiency 3/π ≈ 0.955 under normality ([Hodges & Lehmann 1956](https://doi.org/10.1214/aoms/1177728261)), using the validation d_z (0.86–1.18) → 13–22 prompts needed. n = 100 was pre-registered, the floor of the [100, 300] range, so planned power is ≈ 1.0.

## Validation results (100 fresh discovery prompts, 1 sample; unsteered median distinct-2 = 0.95)
| unit | concept (top contexts) | α | mean ΔC | d_z | median ΔNLL | steered distinct-2 | eligible |
|---|---|---|---|---|---|---|---|
| L19 N3901 − | programming conditionals ("the if statement", "if-else") | 4 | 0.330 | 2.52 | −0.011 | **0.16** ("ififif…") | no (degenerate) |
| L0 N2848 + | Fibonacci / combinatorics paths | 4 | 0.201 | 2.33 | −0.118 | **0.04** ("Fib Fib Fib…") | no (degenerate) |
| L18 N3066 − | phonetics, letters, sounds | 1 | 0.210 | 1.18 | 0.605 | 0.91 | yes |
| L13 N1680 + | non-duality, dependent origination, the Self | 1 | 0.152 | 1.10 | 0.520 | 0.89 | yes |
| L4 N2104 + | pedagogy / pedagogical content knowledge | 2 | 0.121 | 1.09 | 0.605 | 0.94 | yes |
| L6 N3344 − | forum "mark as accepted/correct answer" | 4 | 0.164 | 1.06 | 0.606 | 0.85 | yes |
| L18 N1735 − | electron orbitals / shells / bonding | 2 | 0.148 | 0.86 | 0.648 | 0.94 | yes |
| L5 N2879 + | keyboard modifier keys / shortcuts | 1 | 0.087 | 0.67 | 0.558 | 0.94 | yes (not in top 5) |
| L3 N454 − | type theory / arrows / morphisms | 1 | 0.092 | 0.70 | 0.451 | 0.87 | yes (not in top 5) |
| L17 N4259 + | arithmetic/geometric sequences | 2 | 0.070 | 0.60 | 0.589 | 0.94 | yes (not in top 5) |

## Amendment made before the confirmatory run, and why
The first screening run used perplexity as the only fluency guard. It surfaced L0 N2848 at α = 4: very high concept score, *lower* perplexity than unsteered text, and output that is an endless "Fib Fib Fib…" loop. Repetition is highly predictable for a language model (in-context copying), so perplexity under the unsteered model fails as a degeneracy guard.

We therefore made these changes:
- added a distinct-2 guard (steered median ≥ 0.6);
- validated the top 10 instead of the top 5;
- moved the guard into both eligibility and the confirmatory decision rule.

All of this happened *before* any confirmatory prompt was used. The first, guard-free validation of 5 units is kept in `screen_initial_validation.json`. The guard also excluded the most interpretable unit, the if-statement neuron L19 N3901−, which at α = 4 degenerates into "ififif…". It is pre-registered as an **exploratory** unit for the follow-ups only, outside the Holm family.

## Learnings
- **Concept score is gameable.** A high concept score with low perplexity can be pure repetition, so a lexical-diversity guard is necessary. This is the same reason AxBench pairs concept scores with fluency ratings ([Wu et al. 2025](https://arxiv.org/abs/2501.17148)).
- **Screening is noisy.** At 25 prompts, 1 sample and 32 tokens, the screening ΔC of a unit can differ substantially from its 100-prompt validation ΔC at the same α: L18 N3066 went from 0.103 (screen) to 0.210 (validation), L19 N3901 from 0.387 to 0.330 (`screen.json`: `candidates[*].screen` vs `validation`). Validation on fresh prompts is what makes the power analysis honest.
- **Plan R and K together.** With permutation nulls, the number of null draws must be chosen together with K, or Holm cannot reject even for perfect neurons.
- **Infrastructure:**
  - Port 8080 was taken by another application, so the LLM server uses 8093.
  - Decoding is CPU-dispatch-bound (~0.3 s per step almost independent of batch size).
  - Batches above available VRAM spill silently into system RAM on Windows, roughly 3× slower.
