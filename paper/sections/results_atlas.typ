== Model sanity and the neuron atlas
Held-out loss falls from 4.03 nats at $T = 1$ to 2.51 at the trained $T = 4$ (perplexity 12.3), and rises slightly when
extrapolating to $T = 5..8$ (2.55–2.56). This matches the behavior described for Ouro @zhu2025ouro.
- *Atlas.* 8M tokens, 72.5 min. No neuron is dead. Median $max|a|$ is 0.64.
- *Peak location.* Strongest activations concentrate in early loops: the share of neurons whose top context occurs at
  loop 1, 2, 3, 4 is 41.0%, 29.7%, 16.7%, 12.7%.
- *Cross-loop similarity.* On 32,256 held-out tokens, the median per-neuron Pearson correlation between loop-1 and loop-4
  activations is 0.54 (IQR 0.43–0.63). It rises with depth from $approx 0$ in layer 0 to 0.59–0.65 in layers 14–23.
- *Convergence.* Adjacent loops converge: median $r(1,2) = 0.65$, $r(2,3) = 0.84$, $r(3,4) = 0.94$, consistent with
  fixed-point dynamics @arroyo2026mechanistic.
- *Top-context overlap* between loop 1 and loop 4 is much lower (Jaccard 0.01–0.19 by layer). This mostly reflects how
  sensitive top-$k$ sets are to near-ties (@fig:loops).

#figure(image("../../figures/loop_structure.png", width: 100%),
  caption: [Loop structure of all 135,168 neurons. Left: share of neurons whose strongest atlas context occurs at each
  loop. Middle: Jaccard overlap of top-20 contexts between loops. Right: median same-token activation correlation
  between loops (held-out text).]) <fig:loops>

== Selection, and a failure mode of the fluency guard <sec:select>
Of 8,000 neuron poles, 21.2% are token detectors. The 200 most coherent of the remaining units were quick-steered.

The two units with the largest validated concept effects, an if-statement neuron (L19 N3901−, $d_z = 2.52$) and a
Fibonacci neuron (L0 N2848+, $d_z = 2.33$), turned out to reach them by degenerating into "ififif…" and "Fib Fib Fib…".
This degenerate text has *lower* perplexity under the unsteered model than normal continuations, so a perplexity guard
alone accepts it. We added a distinct-2 guard (steered median $>= 0.6$; unsteered $approx 0.95$) before any
confirmatory prompt was used, and recorded the amendment in the pre-registration.

