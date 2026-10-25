== Replication on Ouro-1.4B-Thinking
The five neuron indices were tested unchanged in Ouro-1.4B-Thinking, which is Ouro-1.4B after supervised fine-tuning on
about 8.3M reasoning examples @zhu2025ouro. There was no re-selection.

Reasoning SFT barely moves the MLP neurons:
- the per-neuron cosine between base and Thinking weights (minimum over gate row, up row and down column) is $>= 0.993$
  for 99% of neurons in the four candidate layers, and $>= 0.996$ for the five candidates;
- the candidates' maximum activations change by at most 5.1% of the base value.

All five neurons steer the post-trained model with similar or larger effects:
- $Delta C = 0.161$–$0.180$, against $0.127$–$0.179$ in the base model (larger for four neurons; for L18 N3066 it is
  0.171 against 0.179, within the confidence intervals);
- Holm-adjusted Wilcoxon $p <= 2.1 times 10^(-16)$;
- each neuron exceeds all 20 random same-layer neurons;
- the PPL ratio is at most 1.98.

With $R = 20$ the smallest permutation $p$ is $1 \/ 21$, which cannot pass Holm at $K = 5$, so the random-neuron
comparison is descriptive here. The code at the start of the run named the Wilcoxon test as the replication criterion;
the "beats all 20" and guard conditions were written down after the run.

A second replication on 100 never-used prompts (the reserve split) gives the same picture:
- $Delta C = 0.157$–$0.179$;
- Holm-adjusted Wilcoxon $p <= 2.0 times 10^(-16)$;
- each neuron again beats all 20 random neurons;
- the PPL ratio is at most 1.97.

#include "results_judge.typ"

= Discussion
*Single neurons are effective steering handles in a looped LM.* Goodfire's essay argues that raw neurons are often
clean, single-purpose units @goodfire2026clean. Our results support a narrower causal version of that claim for a looped
model. Neurons chosen for coherent top contexts include units that steer generation toward their concept, and they
do so far beyond random neurons from the same layer clamped the same way. The effect survives a change of embedding
model and reasoning post-training, and it is specific to each neuron's own concept.

*Looping makes steering cumulative.* A clamp applied at one loop is largely undone by the remaining loops. Only
repeated injection, at every pass through the shared weights, changes behavior. Steering a neuron at all four loops
works well, while for four of the five confirmed neurons any single loop is nearly inert (for L13 N1680, single-loop
effects are small but non-zero). Mechanistically, an injection at loops 1–3 keeps only 33–78% of its size by the
time it reaches the output. Behaviorally this matches the last-loop advantage reported for residual directions in
Ouro-1.4B-Thinking @nesiacel2026recurrent, but the mechanism differs: there, early perturbations persist and are rotated
away from the concept direction, while here the injected displacement itself shrinks. It is also consistent with the
convergence of activations across loops (median same-token activation correlation between consecutive loops rises from
0.65 to 0.94).

The exception, the if-statement neuron, acts mainly at the last loop. That suggests the location of the effect depends
on the neuron: some concepts are written late, close to the output, while others are maintained throughout.

*The neuron's function is mostly, but not entirely, shared across loops.* Same-token activations at loop 1 and loop 4
correlate at a median of 0.54. Neurons are therefore partly reused for different inputs at different depths, as the
input distribution of every layer shifts from loop to loop. Explanations written by a local 9B model do not detect this
difference (same-loop 0.60 vs cross-loop 0.59). With explainers this weak, activation statistics are the more sensitive
instrument.

*Fluency guards must catch repetition.* Concept scores can be maximized by degenerate repetition, which a language model
finds *easy* to predict. Perplexity-based fluency guards can therefore pass degenerate outputs. A lexical-diversity
guard, or an LLM fluency judge as in AxBench @wu2025axbench, is needed alongside them.

= Limitations
- *Scope.* One looped model family (Ouro-1.4B and its Thinking variant) and 5 confirmed neurons, chosen for steerability
  from a pool of 200 coherent units. These are existence claims and say nothing about how common steerable neurons are.
  Ouro-2.6B (48 looped layers) was not attempted: its bf16 checkpoint is 5.3 GB @ouro26bcard and its KV cache per
  token is twice Ouro-1.4B's, which leaves too little of the 8 GB GPU for the batched runs used here.
- *Metric.* The primary metric is an embedding-based concept score built from each neuron's own top contexts. It
  measures movement toward that contrastive direction, not human-judged concept presence. The LLM judge is a secondary
  check from a small local model.
- *Intervention.* We clamp the neuron at every position, including the prompt, and far beyond its natural activation
  pattern ($alpha$ up to 4 for confirmation). Steering costs fluency (perplexity ratio 1.6–1.8).
- *Null resolution.* With $R = 100$ random neurons and $K = 5$, the permutation test's Holm-adjusted $p$ cannot go
  below 0.0495. We report z-margins (7–12 SD) as descriptive support.
- *Post hoc analyses.* Follow-ups use 60 prompts and 1 sample. Ablation $p$-values are unadjusted. The T sweep beyond
  $T = 4$ is extrapolation.
- *Atlas size.* 8M tokens is far smaller than the 100M tokens used by Goodfire, so rare features are under-sampled.
- *Compute.* Everything ran on one 8 GB laptop GPU. The labeler is a 4-bit 9B model.

= Conclusion
In Ouro-1.4B, a looped transformer that applies its 24 layers four times, we built a 135k-neuron atlas and ran a
pre-registered, paired steering test with random same-layer neuron nulls. All five pre-registered neurons (phonetics,
non-duality, pedagogy, forum answers, electron orbitals) steer generation toward their concept with large,
statistically significant effects. They are specific to their own concepts, level off with clamp strength for four of
six neurons (qualitatively consistent with the sandwich-norm bound), and replicate in the reasoning-tuned
Ouro-1.4B-Thinking.

Steering a neuron in a looped model requires clamping it at every loop. Single-loop interventions are mostly undone by
the following loops. Neurons are largely reused across loops, but not identically: their same-token activation
correlation between the first and last loop has a median of 0.54.
