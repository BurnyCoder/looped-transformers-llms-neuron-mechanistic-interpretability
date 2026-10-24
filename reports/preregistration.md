# Pre-registration: confirmatory neuron-steering test on ByteDance/Ouro-1.4B

Written 2026-10-24 22:06:25 by `run_pipeline.py --phase preregister`, **before** any confirmatory prompt was used.
Machine-readable copy: `results/Ouro-1.4B/candidates.json`. Methods: [docs/methodology.md](../docs/methodology.md).

## Hypothesis H1 (primary)
At least one of the K = 5 candidate neurons below, clamped to `alpha x max|a|` on its pole at every position and
all 4 loops, increases the concept score of 48-token continuations relative to (a) unsteered generation and (b) the
random same-layer neuron null.

## Frozen design
- Model `ByteDance/Ouro-1.4B` @ `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, T = 4 loops, exit at the last loop, bf16.
- Prompts: the first **n = 100** stems of `data/prompts/confirm.jsonl` (WikiText-103 test, 12 tokens; never used before).
- k = 3 samples per prompt, 48 new tokens, temperature 0.7, top-p 0.9, seed base 2234;
  steered and unsteered generations share seeds and batch layout (paired).
- Primary metric: per-prompt mean over k of Delta C, C = cosine(bge-small-en-v1.5 embedding of the continuation, the
  neuron's contrastive concept direction computed from its top-20 atlas contexts minus 2000 random atlas contexts).
- Test 1: one-sided Wilcoxon signed-rank on per-prompt Delta C (H1: > 0).
- Test 2: random-neuron null: R = 100 alive neurons from the candidate's layer, clamped at the same alpha x their
  own max on the same pole; statistic = mean Delta C over the first 50 prompts, sample 0, scored with the candidate's
  concept direction; p = (1 + #null >= observed)/(R + 1).
- Multiplicity: Holm across the K candidates, separately for Test 1 and Test 2, family-wise alpha = 0.05
  (smallest attainable Test-2 p = 1/(R+1) = 0.0099 <= 0.05/K = 0.0100).
- **Decision rule:** a candidate is a "significant steering neuron" iff Holm p (Test 1) < 0.05 AND Holm p (Test 2) < 0.05
  AND median per-prompt continuation-perplexity ratio (steered / unsteered, both scored by the unsteered model) <= 2
  AND median distinct-2 of steered continuations >= 0.6 (non-degenerate text).
- Secondary (reported, not part of the decision): all-MiniLM-L6-v2 concept score, distinct-2 change, LLM judge
  (AxBench concept/fluency 0-2), dose-response over alpha in {0.5, 1, 2, 4, 8, 16}, zero-ablation, loop-restricted steering.
- Falsification: if no candidate meets the decision rule, H1 is rejected for this protocol. At most one further
  pre-registered round may be run on `data/prompts/reserve.jsonl`.

## Candidates (selected and validated on discovery prompts only)
Selection: 8,000 units (4,000 random alive neurons x 2 poles) -> token-detector filter -> 200 most coherent -> quick
steer (discovery prompts 1-25, alpha in {1, 2, 4}) -> top 10 by screen Delta C -> validation on
discovery prompts 26-125 -> eligible if mean Delta C > 0, median Delta NLL <= log 2, median distinct-2 >= 0.6
-> the K = 5 eligible units with the largest validation d_z.
Power: required n per candidate for one-sided Wilcoxon (t-test power / ARE 0.955) at alpha = 0.05/K, power 0.9, using the
validation d_z (100 fresh discovery prompts, 1 sample).

| key | layer | neuron | pole | alpha | coherence | validation mean Delta C | validation d_z | validation distinct-2 | required n | power at n | label (local Qwen3.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 18_3066_1 | 18 | 3066 | - | 1.0 | 0.619 | 0.2100 | 1.18 | 0.91 | 13 | 1.00 | The highlighted tokens consistently identify core phonetic or orthographic units—such as vowels, consonant categories, letters, and specific words—that serve as the primary subjects of linguistic analysis regarding pronunciation, sound modification, and structural definition. |
| 13_1680_0 | 13 | 1680 | + | 1.0 | 0.634 | 0.1518 | 1.10 | 0.89 | 15 | 1.00 | The text latents consistently emphasize metaphysical concepts of non-duality, dependent origination, and the nature of the Self, frequently using negation to distinguish the eternal Self from transient phenomena like the body or mind. Key patterns include the assertion that all things arise dependently, the definition of the Self as a pure witness or consciousness beyond attributes, and the dissolution of personal identity into a singular, undifferentiated reality, with highlighted tokens often marking states of being or identity markers central to these arguments. |
| 4_2104_0 | 4 | 2104 | + | 2.0 | 0.659 | 0.1206 | 1.09 | 0.94 | 15 | 1.00 | The text latents commonly describe the domain of pedagogical content knowledge, emphasizing the integration of subject matter expertise with teaching methods in academic and professional contexts. |
| 6_3344_1 | 6 | 3344 | - | 4.0 | 0.638 | 0.1637 | 1.06 | 0.85 | 16 | 1.00 | The text latents describe interface elements and actions for validating and rewarding content quality in a community forum, specifically focusing on marking answers as correct or accepted, upvoting posts, and awarding community recognition like KUDOS or brainliest. |
| 18_1735_1 | 18 | 1735 | - | 2.0 | 0.650 | 0.1482 | 0.86 | 0.94 | 22 | 1.00 | The text latents commonly describe core quantum mechanical concepts related to atomic structure and chemical bonding, specifically focusing on the terminology for electron containers (orbitals, shells, subshells), the particles themselves (electrons), and their energetic or spatial configurations (valence, outermost, filled, hybridization). The patterns indicate a strong focus on defining how electrons are arranged to achieve stability and how specific orbitals participate in forming chemical bonds. |

## Exploratory unit (outside the confirmatory family; no p-value from it enters the Holm family)
| key | layer | neuron | pole | alpha (follow-ups) | why |
|---|---|---|---|---|---|
| 19_3901_1 | 19 | 3901 | - | 2.0 (plus the full dose-response grid) | Most interpretable screened unit: top contexts are programming conditionals ("the if statement", "if-else", "loops, ifs"). At alpha=4 its validation continuations degenerate ("ififif...", distinct-2 0.16), so the guard excluded it from K. It is studied only in the secondary follow-ups (dose-response, loop locus, T sweep, ablation), which are labeled exploratory. |

Amendment log: the distinct-2 guard was added after the screening run (21:36) showed a degenerate high-concept-score unit ("Fib Fib Fib..."), and before any confirmatory prompt was used. The initial 5-candidate validation without the guard is kept in `results/Ouro-1.4B/screen_initial_validation.json`.
