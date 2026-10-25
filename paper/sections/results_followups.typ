== Follow-up experiments (post hoc)
These analyses use 60 held-out prompts, 1 sample each.

*Specificity.* Steering neuron $i$ and scoring concept $j$ gives a matrix whose diagonal is the largest entry of each row,
1.4–3.7× the largest off-diagonal entry (@fig:spec). The smallest margin pairs the forum "accepted answer" neuron with
the programming concept, an overlap that is plausible for Q&A text.

*Dose-response.* $Delta C$ rises with $alpha$ and then plateaus, as the sandwich-norm bound predicts (dips of at most
0.011, inside the CIs, make the rise not strictly monotone).
Neurons with larger $c^* \/ max|a|$ (2.2–2.6) keep rising up to $alpha = 8$–$16$; those near 1 plateau from
$alpha approx 2$–$4$. Clamping to the opposite sign does not steer toward the concept ($|Delta C| <= 0.04$; four of twelve conditions have
a small negative effect whose CI excludes 0), so each concept lives on one pole.
Perplexity ratio exceeds 2 for most neurons at $alpha >= 4$.

*Loop locus (H2).* Clamping a single loop barely steers: $Delta C <= 0.043$, against 0.13–0.17 for all four loops, for
the five confirmed neurons. The four single-loop effects sum to −2% to 30% of the all-loop effect for four of them (76% for L13 N1680).
- Steering in a looped model is therefore strongly super-additive across loops. One injection is largely undone by the
  following loops, consistent with self-repair @nesiacel2026recurrent @mcgrath2023hydra.
- The exploratory if-statement neuron is the exception. Its last loop alone gives 56% of the full effect
  ($Delta C = 0.176$, $p = 1.7 times 10^(-10)$), while loops 1–3 give nothing, matching the last-loop advantage reported
  for residual-direction steering of Ouro @nesiacel2026recurrent.
- Clamping at loop 4 moves the final hidden state most (relative displacement 0.20–0.47, against 0.12–0.23 at loops 1–3).

*Test-time loops (H3).* Clamping at every loop while running $T = 1..8$:
- the concept effect is already present at $T = 1$ ($Delta C$ 0.10–0.35);
- it does not grow systematically with $T$: for the confirmed neurons, $Delta C$ at $T = 1$ is 70–99% of its $T = 4$
  value, and from $T = 2$ to 8 it fluctuates between 0.11 and 0.25 without a monotone trend;
- next-token KL grows with $T$, apart from small dips ($<= 0.06$) for three of six neurons.

More loops add more injections and more distribution shift, but not proportionally more concept.

*Necessity.* Zero-ablating a neuron on its held-out top contexts increases the next-token loss beyond all 100 random
same-layer ablations for 4 of 6 neurons (unadjusted $p = 0.0099$; $z$ = 3.1–53.6). This is exploratory; correcting
with Holm across six would give 0.059. Two confirmed neurons are sufficient for steering but not necessary for
prediction, and ablating L6 N3344 even *improves* prediction ($z = -8.9$).

#figure(image("../../figures/specificity.png", width: 70%),
  caption: [Steering specificity: mean $Delta C$ on each concept (columns) when steering each neuron (rows). Diagonal
  entries dominate their rows.]) <fig:spec>

== Interpretability audit (H4)
Delphi explanations written by the local Qwen3.5-9B from 12 examples per loop detect held-out activating contexts above
chance (loop-4 explanations: balanced accuracy 0.60 vs 0.50 for shuffled explanations on the same contexts, paired
difference +0.097, two-sided Wilcoxon $p = 1.4 times 10^(-7)$, $n = 123$). However, they transfer across loops as well
as within a loop (same 0.60, cross 0.59; paired difference +0.012, two-sided Wilcoxon $p = 0.20$, $n = 240$).

H4 (loop-specific explanations) is therefore not supported. Only 25% (loop 1, Wilson 95% CI 18–33%) and 21% (loop 4,
CI 15–28%) of neurons reach balanced accuracy $>= 0.75$,
so the weak explainer limits how sensitive this test can be, and the activation correlations above are the more
sensitive evidence.

== Robustness and mechanism (plan-completion analyses, post hoc)
- *Seed sensitivity.* Re-running the confirmatory comparison with three new seed sets (100 prompts, 1 sample each)
  gives $Delta C = 0.11$–$0.18$ for every neuron in every set ($p <= 2.9 times 10^(-8)$). The spread across seed sets is
  0.007–0.024.
- *Keyword-probability shift.* Teacher-forced on unsteered continuations, clamping a neuron raises the next-token
  probability mass on its keyword set by 0.45–0.94 nats on average ($approx 1.6$–$2.6 times$, all
  $p approx 2 times 10^(-18)$). The keywords are up to 20 tokens over-represented around its atlas peaks, e.g. "sounds,
  vowel" for phonetics, "pedagogy, instruction" for pedagogy, "orbital, shell, outermost" for orbitals. The lists also
  contain function words and punctuation (the non-duality neuron's list is almost only function words), so this is a
  peak-context vocabulary rather than a curated concept vocabulary.
- *Self-repair ratio.* A clamp at loop 1, 2 or 3 leaves 33–78% of its immediate residual displacement at the output
  (six neurons, including the exploratory one), while a clamp at loop 4 leaves 67–126% (above 100% for four of six). Later loops thus actively shrink earlier injections, quantifying the
  super-additivity above.
- *Silencing.* Holding a neuron at 0 during generation does not change the concept score ($|Delta C| <= 0.005$).
- *Saturation in $T$.* Next-token KL saturates with the number of loops ($tau approx 2$–$4$ loops, $R^2 = 0.90$–$0.99$, five confirmed neurons),
  while the concept effect is already near its plateau at $T = 1$–$2$.
- *Positional neurons.* None of the 10 validated candidates is positional (no position holds more than 3 of the 40 top
  peaks, i.e. 7.5%).
