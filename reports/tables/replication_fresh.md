| neuron (layer, index, pole) | alpha | concept (Qwen3.5 label, shortened) | mean Delta C [95% CI] | d_z | rank-biserial | p Wilcoxon (Holm) | p random-neuron null (Holm) | null mean / q95 | PPL ratio | steered distinct-2 | MiniLM Delta C | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L18 N3066 - | 1.0 | The highlighted tokens consistently identify core phonetic or orthographic units—such as v | 0.175 [0.152, 0.199] | 1.46 | 0.97 | 1.4e-17 (2.8e-17) | 0.0476 (0.2381) | -0.004 / 0.005 | 1.84 | 0.89 | 0.247 | not significant |
| L13 N1680 + | 1.0 | The text latents consistently emphasize metaphysical concepts of non-duality, dependent or | 0.179 [0.154, 0.203] | 1.44 | 1.00 | 2.6e-18 (1.1e-17) | 0.0476 (0.2381) | 0.014 / 0.026 | 1.82 | 0.90 | 0.227 | not significant |
| L4 N2104 + | 2.0 | The text latents commonly describe the domain of pedagogical content knowledge, emphasizin | 0.165 [0.147, 0.182] | 1.80 | 1.00 | 2.1e-18 (1.1e-17) | 0.0476 (0.2381) | -0.011 / 0.005 | 1.82 | 0.90 | 0.224 | not significant |
| L6 N3344 - | 4.0 | The text latents describe interface elements and actions for validating and rewarding cont | 0.168 [0.150, 0.188] | 1.69 | 0.99 | 3.7e-18 (1.1e-17) | 0.0476 (0.2381) | 0.002 / 0.052 | 1.88 | 0.77 | 0.123 | not significant |
| L18 N1735 - | 2.0 | The text latents commonly describe core quantum mechanical concepts related to atomic stru | 0.157 [0.132, 0.184] | 1.18 | 0.94 | 2.0e-16 (2.0e-16) | 0.0476 (0.2381) | 0.006 / 0.016 | 1.97 | 0.89 | 0.217 | not significant |

The decision column applies the confirmatory rule. With R = 20 random neurons and K = 5, the smallest attainable Holm-adjusted null p is 0.238, so it reads 'not significant' by construction; the replication criterion is in reports/06_replication.md.
