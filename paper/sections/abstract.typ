#block(inset: (x: 1.2em))[
*Abstract.* Looped language models apply the same layers several times per token, so every MLP neuron fires once per
loop. We study the 135,168 SwiGLU neurons of Ouro-1.4B, which loops its 24 layers four times, at the level of single
neurons, following Goodfire's raw-neuron methodology.

*Atlas.* From 8M tokens we build per-loop top-context tables.
- Strongest activations concentrate in early loops (41% at loop 1, 13% at loop 4).
- Same-token activations of a neuron at loops 1 and 4 correlate with median $r = 0.54$, and consecutive loops converge
  ($r$: 0.65, 0.84, 0.94).

*Pre-registered test.* We selected candidates on discovery prompts only and committed a pre-registration before using
held-out prompts. Clamping a single neuron at all loops was compared against unsteered generation and against 100 random
neurons from the same layer, with Holm correction and fluency and degeneracy guards.
- *All five* pre-registered neurons (phonetics, non-duality, pedagogy, forum answers, electron orbitals) steer 48-token
  continuations toward their concept: mean $Delta$concept-score 0.13–0.18, $d_z$ 0.99–1.61, Holm-adjusted Wilcoxon
  $p <= 1.1 times 10^(-15)$, beating 99–100 of 100 random neurons, perplexity ratio $<= 1.83$.
- The effects are concept-specific, confirmed by a blind LLM judge, and replicate at identical indices in the
  reasoning-tuned Ouro-1.4B-Thinking.

*Follow-ups (post hoc).*
- Clamping at any single loop barely steers ($Delta C <= 0.043$ versus 0.13–0.17 for all loops), so steering is strongly
  super-additive across loops, consistent with self-repair.
- Dose-response saturates, as predicted by the post-MLP RMSNorm.

*Negative results.*
- Perplexity-based fluency guards accept degenerate repetitions ("Fib Fib Fib…").
- Explanations from a local 9B model transfer across loops as well as within them, so this audit could not detect
  loop-specific function.
]
#v(6pt)
