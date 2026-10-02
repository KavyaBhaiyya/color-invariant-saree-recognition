Test gallery size: 327

| Identification (synthetic recolour query vs original gallery) | Top-1 | Top-5 | Top-10 |
|---|---|---|---|
| all test images | 0.4220 | 0.9419 | 0.9817 |
| natural_identification_any_colour (n_queries=274) | 0.9781 | 1.0000 | 1.0000 |
| natural_identification_different_colour | n/a (no such pairs) | | |

| Verification | ROC-AUC | EER | TPR@1%FPR | n_pos | n_neg |
|---|---|---|---|---|---|
| synthetic_recolor_pos_vs_colour_matched_neg | 0.9944 | 0.0306 | 0.9450 | 327 | 327 |
| synthetic_recolor_pos_vs_random_neg | 0.9945 | 0.0321 | 0.9450 | 327 | 327 |
| natural_same_group_vs_colour_matched_neg | 0.9897 | 0.0229 | 0.4934 | 458 | 458 |
| natural_same_group_different_colour_vs_colour_matched_neg | n/a | | | | |

Params: 24.03 M | embedding dim: 256 | FLOPs: 8.18 G/image | latency (cuda, fp32): 5.7 ms (bs1), 84.9 ms (bs32)