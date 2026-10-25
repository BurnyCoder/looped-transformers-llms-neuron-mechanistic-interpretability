= Method <sec:method>

== Model and neuron definition
We study Ouro-1.4B @zhu2025ouro at revision `574fa66`. It has 24 decoder layers (d = 2048, 16-head MHA,
SwiGLU MLP of width 5632, vocabulary 49,152). The whole stack is applied $T = 4$ times with shared weights. Each layer uses
sandwich RMSNorm, and the residual stream is RMS-normalized after every loop. We always read the output of the last loop
(`exit_at_step = T-1`). The default early-exit path, which thresholds a cumulative exit distribution, disagreed with
fixed last-loop exit on 0 of 2,048 held-out positions (8 chunks × 256), but fixing the exit makes every computation identical by construction.
A *neuron* $(ell, i)$ is element $i$ of the SwiGLU activation entering `down_proj` in layer $ell$:
$ a_(ell,i,t) = "silu"(g_i^top x) dot (u_i^top x) $
Here $x$ is the post-attention-normalized residual at loop $t$. There are 24 × 5632 = 135,168 neurons, and each fires
once per loop. Activations are signed, so following
@goodfire2026clean we treat the positive and negative *poles* of every neuron as separate units. Reading and editing
use one forward pre-hook per `down_proj`; the loop index is the per-layer call count modulo $T$.
Unit tests check the hook against a manual SwiGLU recomputation at every loop, confirm that identity edits leave the
logits bit-identical, and confirm that cached decoding matches full forward passes.

== Neuron atlas
We stream 8M tokens: FineWeb-Edu @penedo2024fineweb (90%) plus OpenWebMath @paster2023openwebmath (10%), packed into
256-token chunks. For every neuron and pole we keep the top-40 contexts, taking the maximum over positions and
loops and recording the loop at which it occurs. We also keep the top-20 contexts at each loop, and per-loop moments and
exceedance counts (computed on every 8th batch). The first 4 positions of each chunk are excluded.

== Candidate selection on discovery data
Prompts are 12-token stems of WikiText-103 *test* paragraphs @merity2016wikitext, a corpus disjoint from the atlas text.
They are split once, with a fixed seed, into discovery (150), confirmatory (300) and reserve (300) sets.

*Selection.* From a random pool of 4,000 alive neurons (8,000 units), we embed each unit's top-20 contexts with
bge-small-en-v1.5 @xiao2023bge and rank units by coherence (mean pairwise cosine). Units whose top contexts share a
single peak token at least 50% of the time are excluded as token detectors @voita2023neurons. The 200 most coherent
units are clamped at $alpha in {1, 2, 4}$ times their maximum activation, on 25 discovery prompts, to obtain a quick
estimate of the effect. The ten best units that pass a perplexity guard are then *validated* on 100 different discovery
prompts, so that the effect sizes entering the power analysis are not inflated by selection. A unit is eligible if its
validation mean $Delta C > 0$, its median perplexity ratio is at most 2, and its median steered distinct-2 is at least
0.6 (a degeneracy guard; see @sec:select). The five eligible units with the largest validation $d_z$ are
pre-registered.

== Steering intervention and primary metric
Steering clamps $a_(ell,i,t) := c = ± alpha dot max|a|$ (sign of the pole) at every position and, unless stated
otherwise, every loop. Because the MLP write is RMS-normalized after `down_proj`, a single clamped neuron's
contribution per layer and loop is bounded. As $|c| -> infinity$ the write tends to $gamma ⊙ "sign"(c) w_i / "rms"(w_i)$,
so the dose-response must saturate (scale $c^* approx ||W_"down" a|| \/ ||w_i||$).

For each prompt we sample $k = 3$ continuations of 48 tokens (temperature 0.7, top-$p$ 0.9). Steered and unsteered runs
share seeds and batch layout, so each steered sample is paired with an unsteered sample.

The *concept score* of a continuation is the cosine between its bge-small embedding and the unit's
*contrastive concept direction* $u = "normalize"(overline(e)_"top" - overline(e)_"random")$. Here $overline(e)_"top"$ is
the mean embedding of the unit's top-20 atlas contexts and $overline(e)_"random"$ the mean over 2,000 random atlas
contexts. The primary per-prompt statistic is $Delta C$, the steered minus unsteered score averaged over the $k$
samples. It involves no LLM, so a label cannot leak into its own evaluation.

== Pre-registered confirmatory test
Candidates, $alpha$, $n$, the tests and the decision rule were written to `reports/preregistration.md` and committed
before any confirmatory prompt was used. The pipeline refuses to run the test otherwise.

- *Test 1:* one-sided Wilcoxon signed-rank on per-prompt $Delta C$.
- *Test 2:* random-neuron null @goodfire2026clean. $R = 100$ alive neurons from the candidate's layer are clamped at the
  same $alpha$ times their own maximum on the same pole. Each is scored with the *candidate's* concept direction on the
  first 50 prompts, giving the empirical $p = (1 + \#{"null" >= "obs"}) \/ (R+1)$ @phipson2010permutation.
- *Multiplicity:* Holm @holm1979 across the $K$ candidates, separately for each test. $K <= 5$ is forced because the
  smallest attainable permutation $p$, $1\/101$, must be at most $0.05\/K$.
- *Decision rule:* a neuron is a significant steering neuron iff both Holm-adjusted $p < 0.05$, the median perplexity
  ratio of steered to unsteered continuations (both scored by the unsteered model) is at most 2, and the median
  distinct-2 of steered continuations is at least 0.6.
- *Effect sizes:* $d_z$ and matched-pairs rank-biserial $r$ @kerby2014rank, with bootstrap 95% CIs.
- *Power:* $n$ is chosen from the validation $d_z$ by power analysis (one-sided paired $t$ at $0.05\/K$, power 0.9,
  divided by the Wilcoxon ARE of 0.955), clipped to [100, 300].

== Secondary analyses (post hoc unless stated)
These analyses are not part of the decision rule:
- an all-MiniLM-L6-v2 concept score;
- distinct-2 @li2016diversity;
- AxBench-style blind LLM ratings @wu2025axbench from a local Qwen3.5-9B @qwen35 served by llama.cpp @llamacpp;
- dose-response over $alpha$;
- zero-ablation of the neuron on its held-out top contexts versus 100 random same-layer ablations;
- *loop locus* (clamping at a single loop);
- *test-time loops* ($T = 1..8$).

The loop structure of all neurons is summarized by:
- the loop holding each neuron's strongest context;
- the Jaccard overlap of top-20 contexts between loops;
- per-neuron Pearson correlation of activations between loops on the same held-out tokens.

The interpretability audit asks the local LLM, using the Delphi explainer and detection scorer @paulo2024delphi, to
describe 150 neurons separately at loop 1 and loop 4, and scores each explanation on held-out contexts from the same and
from the other loop.
