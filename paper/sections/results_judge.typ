== LLM judge (secondary, pre-registered)
A local Qwen3.5-9B rated continuations of 50 confirm prompts using the AxBench concept and fluency templates
@wu2025axbench. Each continuation was rated by itself, in a shuffled order, so the judge was blind to condition.

*Concept presence (0–2).* The neuron's Qwen label served as the concept, and ratings rose from $approx 0$ unsteered to:
- 0.26 for L18 N3066−;
- 0.24 for L13 N1680+;
- 0.26 for L4 N2104+;
- 0.36 for L18 N1735−;
- 0.06 for L6 N3344−.

Paired one-sided Wilcoxon $p$ = 0.003, 0.0007, 0.0007, 0.0001 and 0.042 respectively. All remain below 0.05 after Holm.

*Fluency (0–2).* Fluency fell from 1.50 to 0.90–1.17 (95% CIs of the paired differences exclude 0), which agrees with
the perplexity cost.

The judge is strict: absolute concept ratings are low, and the forum-answer neuron (L6 N3344), whose concept is the most
abstract, is the weakest by this measure. The ranking of neurons only partly matches the embedding metric.
