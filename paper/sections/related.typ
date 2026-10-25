= Related work
*Looped and recurrent-depth language models.* The models differ mainly in what they loop and how they inject the input:
- Huginn loops a middle block, with input injection and a randomly initialized latent state @geiping2025huginn.
- Ouro loops the whole stack and learns an early-exit gate @zhu2025ouro.
- Parcae stabilizes looping by constraining the spectral norm of the injection @prairie2026parcae.

On the mechanistic side, Blayney et al. @blayney2026mechanistic find that recurrent blocks converge to cyclic fixed points, at which
attention-head behavior stabilizes, and that they repeat feedforward-like stages of inference. Blog-length studies
intervene on Ouro with residual-stream directions. One finds a last-loop steering advantage and apparent self-repair in
Ouro-1.4B-Thinking @nesiacel2026recurrent. Another finds that per-pass refusal directions in Ouro-2.6B converge from the
second pass onward and that projecting them out removes refusal @suiche2026abliterating. A study of a frozen 2.6B looped
model (Ouro-RLTT) finds that internal readouts give validated decision-level gains, but that directional steering is
negative and generative control yields no gain @kirin2026introspection. Self-repair after ablations is also known in feedforward models
@mcgrath2023hydra.

*Neuron-level interpretability.* There are three strands relevant here:
- Explaining neurons from max-activating examples, possibly with an LLM: @bills2023language, Delphi @paulo2024delphi,
  which also introduced the detection scorer we use, and Transluce's descriptions of every Llama-3.1-8B-Instruct MLP
  neuron @choi2024transluce. Output-centric descriptions, which describe a feature by its effect on outputs (vocabulary
  projection, or the tokens promoted when the feature is stimulated), complement activation-based ones @gurarieh2025output.
- Studies of specific neuron families: universal @gurnee2024universal, confidence-regulation @stolfo2024confidence,
  dead/n-gram/positional @voita2023neurons, sparse-probing @gurnee2023haystack, vocabulary-promoting value vectors
  @geva2022ffn, and gated (SwiGLU) neurons analysed through their input-output weight interaction @gerstner2025gated.
- Goodfire's essay @goodfire2026clean, which argues that most neuron poles of modern models have one main job. It reads,
  steers and ablates raw neurons, and compares ablations to random neurons from the same layers. We adopt that design,
  extend the random same-layer control to steering, and make it quantitative with pre-registered tests.

*Steering evaluation.* Activation addition @turner2023actadd is the basic residual-stream steering method. AxBench
@wu2025axbench rates steered text with an LLM judge for concept presence, instruction relevance and fluency (0–2
each); we use its concept and fluency prompts as a secondary metric. Our primary metric is LLM-free, so explanations cannot leak into their own evaluation.

*Methodology.* We follow guidance on avoiding ML pitfalls @lones2021pitfalls (keep test data out of model selection;
report statistical significance carefully) and the NeurIPS paper checklist @neurips2026checklist (error bars and
statistical significance). Pre-registration, multiplicity control and reporting negative results are our own
additions.
