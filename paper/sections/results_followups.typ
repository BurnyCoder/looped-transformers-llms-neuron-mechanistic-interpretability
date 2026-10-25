== Follow-up experiments (post hoc)
These analyses use 60 held-out prompts, 1 sample each.

*Specificity.* Steering neuron $i$ and scoring concept $j$ gives a matrix whose diagonal is the largest entry of each row,
1.4–3.8× the largest off-diagonal entry (@fig:spec). The smallest margin pairs the forum "accepted answer" neuron with
the programming concept, an overlap that is plausible for Q&A text.

*Dose-response.* $Delta C$ rises monotonically with $alpha$ and then plateaus, as the sandwich-norm bound predicts.
Neurons with larger $c^* \/ max|a|$ (2.2–2.6) keep rising up to $alpha = 8$–$16$; those near 1 plateau from
$alpha approx 2$–$4$. Clamping to the opposite sign has no effect ($|Delta C| <= 0.04$), so each concept lives on one pole.
Perplexity ratio exceeds 2 for most neurons at $alpha >= 4$.

*Loop locus (H2).* Clamping a single loop barely steers: $Delta C <= 0.043$, against 0.13–0.17 for all four loops, for
the five confirmed neurons. The four single-loop effects sum to 0–30% of the all-loop effect for four of them.
- Steering in a looped model is therefore strongly super-additive across loops. One injection is largely undone by the
  following loops, consistent with self-repair @nesiacel2026recurrent @mcgrath2023hydra.
- The exploratory if-statement neuron is the exception. Its last loop alone gives 56% of the full effect
  ($Delta C = 0.176$, $p = 1.7 times 10^(-10)$), while loops 1–3 give nothing, matching the last-loop advantage reported
  for residual-direction steering of Ouro @nesiacel2026recurrent.
- Clamping at loop 4 moves the final hidden state most (relative displacement 0.20–0.47, against 0.12–0.23 at loops 1–3).

*Test-time loops (H3).* Clamping at every loop while running $T = 1..8$:
- the concept effect is already present at $T = 1$ ($Delta C$ 0.10–0.35);
- it stays roughly flat in $T$;
- next-token KL grows monotonically with $T$.

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
chance (balanced accuracy 0.60 vs 0.50 for shuffled explanations, $p = 1.4 times 10^(-7)$). However, they transfer
across loops as well as within a loop (same 0.60, cross 0.59; paired difference +0.012, $p = 0.20$, $n = 240$).

H4 (loop-specific explanations) is therefore not supported. Only about 20% of neurons reach balanced accuracy $>= 0.75$,
so the weak explainer limits how sensitive this test can be, and the activation correlations above are the more
sensitive evidence.
