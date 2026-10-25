= Introduction
Looped (recurrent-depth) language models apply one block of layers several times in sequence, which lets them spend
more computation per token without adding parameters @geiping2025huginn @zhu2025ouro @prairie2026parcae. Ouro-1.4B
@zhu2025ouro loops its entire 24-layer stack four times. Its model card reports performance matching
3–4B-parameter standard transformers @ouro14bcard. Mechanistic work on looped models has so far studied the residual stream:
- recurrent blocks converge to cyclic fixed points and repeat the "stages of inference" of feedforward models @blayney2026mechanistic;
- residual-direction steering of Ouro-1.4B-Thinking works best at the last loop, and earlier injections are
  "rotated away" @nesiacel2026recurrent;
- per-execution-step refusal directions in Ouro-2.6B align from the second pass onward (a follow-up to a study of
  Nanbeige4.2-3B) @suiche2026abliterating.

No published study we are aware of works at the level of *individual neurons* in a looped model.

Neurons are a natural unit for looped models for two reasons. First, a recent essay from Goodfire argues that most MLP
neurons of modern models are already interpretable without a learned decomposition @goodfire2026clean. Second, weight
sharing creates a test that has no analogue in a feedforward network: the same neuron, with the same weights, is
evaluated at every loop. It may do the same job each time, or it may be reused for different jobs at different depths.

We ask three questions:
1. What do Ouro's neurons respond to, and at which loop?
2. Does a neuron keep its function across loops?
3. Can clamping a *single* neuron change what the model writes, with statistical significance against pre-registered
   controls, including random neurons from the same layer clamped the same way?

*Contributions.*
- A neuron atlas covering all 135,168 Ouro-1.4B neurons, with per-loop top contexts and statistics.
- A pre-registered, paired steering test with permutation nulls built from random same-layer neurons, a multiplicity
  correction, a fluency guard, and power analysis on selection-free validation data.
- Measurements of loop structure: where neurons peak, and how similar their activations are across loops.
- Loop-resolved causal experiments: single-loop steering, test-time loop sweeps, and the saturation predicted by the
  sandwich norm.
- An interpretability audit of explanation transfer across loops.

Positive and negative results are both reported.
