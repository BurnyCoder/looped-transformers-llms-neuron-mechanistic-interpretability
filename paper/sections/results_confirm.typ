== Confirmatory test (H1, pre-registered)
@tab:confirm reports the pre-registered test.

*All five candidates meet the decision rule, so H1 is supported.*
- Wilcoxon $p <= 1.1 times 10^(-15)$ before correction for every neuron.
- Every bootstrap 95% CI of mean $Delta C$ excludes zero.
- Each neuron's 50-prompt statistic exceeds all 100 (four neurons) or 99 of 100 (L4 N2104) random same-layer neurons clamped
  the same way. The candidates sit 7.3–12.1 null standard deviations (population SD) above the null mean.
- The Holm-adjusted permutation $p = 0.0495$ is the smallest value attainable with $K = 5$ and $R = 100$. It reflects
  the resolution of the null, not a marginal effect.
- Effects are large: $d_z = 0.99$–$1.61$ and matched-pairs rank-biserial $r = 0.91$–$0.99$.
- The secondary MiniLM embedder agrees ($Delta C = 0.108$–$0.248$).
- Steering costs fluency but stays within the guard: median perplexity ratio 1.58–1.83, and steered distinct-2 is
  0.80–0.91 (unsteered text had $approx 0.95$ in the discovery validation run).

#figure(
  table(
    columns: 9,
    align: (left, center, left, center, center, center, center, center, center),
    stroke: 0.4pt,
    [*Neuron*], [$alpha$], [*Concept (top contexts)*], [*mean $Delta C$ [95% CI]*], [$d_z$], [*Holm $p_"Wil"$*], [*$p_"null"$ (Holm)*], [*z vs null*], [*PPL ratio*],
    [L18 N3066 −], [1], [phonetics, letters], [0.179 [0.155, 0.204]], [1.42], [2.4e-17], [0.0099 (0.0495)], [12.1], [1.58],
    [L13 N1680 +], [1], [non-duality, dependent origination], [0.141 [0.119, 0.163]], [1.26], [1.7e-16], [0.0099 (0.0495)], [11.3], [1.62],
    [L4 N2104 +], [2], [pedagogy], [0.130 [0.112, 0.148]], [1.43], [2.5e-17], [0.0198 (0.0495)], [7.6], [1.65],
    [L6 N3344 −], [4], [forum "mark as accepted answer"], [0.167 [0.147, 0.188]], [1.61], [2.4e-17], [0.0099 (0.0495)], [7.3], [1.83],
    [L18 N1735 −], [2], [electron orbitals, bonding], [0.127 [0.103, 0.153]], [0.99], [1.1e-15], [0.0099 (0.0495)], [11.2], [1.72],
  ),
  caption: [Pre-registered confirmatory test on 100 held-out prompts × 3 paired samples. $Delta C$: change in concept
  score (bge-small). $p_"null"$: empirical p against 100 random same-layer neurons. z vs null is descriptive (post hoc).
  All five rows meet the decision rule. Source: `results/Ouro-1.4B/confirm.json`.],
) <tab:confirm>

#figure(image("../../figures/confirm_effects.png", width: 95%),
  caption: [Mean $Delta C$ (100 prompts × 3 samples) with 95% bootstrap CIs (blue) and the 95th percentile of the
  random same-layer neuron null (orange tick; the null statistic uses the first 50 prompts, sample 0) for each
  candidate.]) <fig:confirm>

Steered text visibly carries the concepts. The examples below are the second confirmatory prompt, sample 0 (chosen from
the first three prompts for visible concept content; the eight stored prompts per neuron are in `confirm_examples.jsonl`):
- L18 N3066− (phonetics) continues "In the Philippines, Nepartak produced strong winds, heavy" with "rains, and severe
  flooding. The sounds of the letters /p/ and /t/ are softened. In the International Phonetic Alphabet (IPA) …".
- The unsteered model continues the same prompt with casualty and damage figures.
- L4 N2104+ (pedagogy) instead writes "rains, and clear skies. - Aarav, a Hindu sage, taught his students through a
  unique method.", then a heading "The Role of an Educator" and "An educator, a dedicated instructor, plays a pivotal …".
