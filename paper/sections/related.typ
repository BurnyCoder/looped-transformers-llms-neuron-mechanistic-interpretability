= Related work
*Looped and recurrent-depth language models.* The models differ mainly in what they loop and how they inject the input:
- Huginn loops a middle block, with input injection and a randomly initialized latent state @geiping2025huginn.
- Ouro loops the whole stack and learns an early-exit gate @zhu2025ouro.
- Parcae stabilizes looping by constraining the spectral norm of the injection @prairie2026parcae.

On the mechanistic side, @arroyo2026mechanistic analyze fixed points and attention-mixing stages. Blog-length studies
steer Ouro with residual-stream directions and find a last-loop advantage, along with apparent self-repair
@nesiacel2026recurrent @suiche2026abliterating. Self-repair after ablations is also known in feedforward models
@mcgrath2023hydra.

*Neuron-level interpretability.* There are three strands relevant here:
- Explaining neurons from max-activating examples, possibly with an LLM: @bills2023language and Delphi @paulo2024delphi,
  which also introduced the detection scorer we use.
- Studies of specific neuron families: universal @gurnee2024universal, confidence-regulation @stolfo2024confidence,
  dead/n-gram/positional @voita2023neurons, sparse-probing @gurnee2023haystack, and vocabulary-promoting value vectors
  @geva2022ffn.
- Goodfire's essay @goodfire2026clean, which argues that most neuron poles of modern models have one main job. It reads,
  steers and ablates raw neurons, and compares ablations to random neurons from the same layers. We adopt that design
  and make it quantitative with pre-registered tests.

*Steering evaluation.* Activation addition @turner2023actadd is the basic residual-stream steering method. AxBench
@wu2025axbench rates steered text for concept presence and fluency with an LLM judge; we use its prompts as a
secondary metric. Our primary metric is LLM-free, so explanations cannot leak into their own evaluation.

*Methodology.* We follow common guidance on avoiding ML pitfalls @lones2021pitfalls: disjoint selection and test data,
pre-registration, multiplicity control, and reporting of negative results.
