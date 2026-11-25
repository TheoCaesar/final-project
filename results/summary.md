# Results — Elliptic, temporal split (train 1–29, val 30–34, test 35–49)

Mean ± sd over seeds. Illicit = positive class. `@val` = threshold chosen on validation F1. pre43/post43 = test steps before/after the dark-market shutdown.


## Features: LF (local 93 only)

| model | n | test_pr_auc | test_roc_auc | test_f1@val | test_precision@val | test_recall@val | test_f1@0.5 | test_f1@val_pre43 | test_f1@val_post43 |
|---|---|---|---|---|---|---|---|---|---|
| RF | 3 | 0.741 ± 0.005 | 0.881 ± 0.003 | 0.709 ± 0.058 | 0.776 ± 0.178 | 0.671 ± 0.046 | 0.724 ± 0.011 | 0.813 ± 0.039 | 0.022 ± 0.003 |
| MLP | 3 | 0.739 ± 0.017 | 0.895 ± 0.002 | 0.755 ± 0.022 | 0.854 ± 0.083 | 0.680 ± 0.021 | 0.464 ± 0.052 | 0.849 ± 0.018 | 0.000 ± 0.000 |
| GCN | 3 | 0.656 ± 0.015 | 0.880 ± 0.004 | 0.595 ± 0.062 | 0.569 ± 0.142 | 0.641 ± 0.032 | 0.275 ± 0.027 | 0.735 ± 0.032 | 0.000 ± 0.000 |
| GAT | 3 | 0.745 ± 0.015 | 0.912 ± 0.008 | 0.747 ± 0.008 | 0.835 ± 0.012 | 0.676 ± 0.020 | 0.411 ± 0.053 | 0.842 ± 0.015 | 0.018 ± 0.001 |
| SAGE | 3 | 0.723 ± 0.023 | 0.899 ± 0.008 | 0.701 ± 0.010 | 0.777 ± 0.058 | 0.642 ± 0.039 | 0.433 ± 0.006 | 0.823 ± 0.020 | 0.007 ± 0.008 |
| DirGNN(a=0.5) | 3 | 0.716 ± 0.008 | 0.898 ± 0.008 | 0.673 ± 0.041 | 0.707 ± 0.091 | 0.646 ± 0.014 | 0.411 ± 0.053 | 0.796 ± 0.013 | 0.000 ± 0.000 |
| Undir-Global | 3 | 0.744 ± 0.005 | 0.906 ± 0.005 | 0.716 ± 0.002 | 0.763 ± 0.014 | 0.675 ± 0.012 | 0.444 ± 0.043 | 0.843 ± 0.005 | 0.002 ± 0.004 |
| CAGNN-style (Undir-Node) | 3 | 0.749 ± 0.010 | 0.904 ± 0.007 | 0.738 ± 0.011 | 0.873 ± 0.064 | 0.641 ± 0.030 | 0.485 ± 0.038 | 0.838 ± 0.017 | 0.002 ± 0.004 |
| Dir-Global | 3 | 0.728 ± 0.021 | 0.910 ± 0.007 | 0.710 ± 0.024 | 0.808 ± 0.061 | 0.637 ± 0.039 | 0.485 ± 0.019 | 0.819 ± 0.025 | 0.000 ± 0.000 |
| NADir (Dir-Node, proposed) | 3 | 0.738 ± 0.044 | 0.914 ± 0.012 | 0.712 ± 0.047 | 0.815 ± 0.022 | 0.636 ± 0.079 | 0.510 ± 0.046 | 0.817 ± 0.052 | 0.013 ± 0.004 |

## Features: AF (all 165)

| model | n | test_pr_auc | test_roc_auc | test_f1@val | test_precision@val | test_recall@val | test_f1@0.5 | test_f1@val_pre43 | test_f1@val_post43 |
|---|---|---|---|---|---|---|---|---|---|
| RF | 3 | 0.764 ± 0.003 | 0.891 ± 0.004 | 0.480 ± 0.085 | 0.359 ± 0.100 | 0.749 ± 0.013 | 0.703 ± 0.031 | 0.647 ± 0.080 | 0.013 ± 0.006 |
| MLP | 3 | 0.504 ± 0.020 | 0.845 ± 0.012 | 0.508 ± 0.012 | 0.486 ± 0.017 | 0.532 ± 0.009 | 0.383 ± 0.012 | 0.576 ± 0.008 | 0.019 ± 0.003 |
| GCN | 3 | 0.418 ± 0.075 | 0.853 ± 0.010 | 0.526 ± 0.036 | 0.525 ± 0.032 | 0.528 ± 0.044 | 0.418 ± 0.053 | 0.605 ± 0.046 | 0.000 ± 0.000 |
| GAT | 3 | 0.429 ± 0.067 | 0.870 ± 0.016 | 0.488 ± 0.059 | 0.481 ± 0.040 | 0.499 ± 0.091 | 0.485 ± 0.037 | 0.557 ± 0.063 | 0.009 ± 0.008 |
| SAGE | 3 | 0.431 ± 0.032 | 0.867 ± 0.014 | 0.526 ± 0.015 | 0.496 ± 0.028 | 0.560 ± 0.008 | 0.483 ± 0.038 | 0.597 ± 0.014 | 0.016 ± 0.004 |
| DirGNN(a=0.5) | 3 | 0.498 ± 0.038 | 0.874 ± 0.005 | 0.505 ± 0.009 | 0.498 ± 0.026 | 0.515 ± 0.026 | 0.497 ± 0.014 | 0.573 ± 0.010 | 0.020 ± 0.004 |
| Undir-Global | 3 | 0.510 ± 0.063 | 0.863 ± 0.024 | 0.548 ± 0.011 | 0.570 ± 0.071 | 0.540 ± 0.074 | 0.503 ± 0.043 | 0.618 ± 0.013 | 0.019 ± 0.007 |
| CAGNN-style (Undir-Node) | 3 | 0.311 ± 0.021 | 0.871 ± 0.009 | 0.476 ± 0.052 | 0.467 ± 0.015 | 0.495 ± 0.110 | 0.428 ± 0.089 | 0.544 ± 0.067 | 0.019 ± 0.010 |
| Dir-Global | 3 | 0.435 ± 0.038 | 0.870 ± 0.016 | 0.505 ± 0.012 | 0.473 ± 0.012 | 0.542 ± 0.026 | 0.479 ± 0.032 | 0.576 ± 0.016 | 0.020 ± 0.001 |
| NADir (Dir-Node, proposed) | 3 | 0.348 ± 0.036 | 0.856 ± 0.003 | 0.480 ± 0.028 | 0.477 ± 0.009 | 0.485 ± 0.048 | 0.467 ± 0.009 | 0.545 ± 0.033 | 0.017 ± 0.004 |

## Mean gate weights on test nodes (averaged over seeds)

| features   | model                      |   layer | class   |   gate_self |   gate_in |   gate_out |   gate_nbr |
|:-----------|:---------------------------|--------:|:--------|------------:|----------:|-----------:|-----------:|
| AF         | CAGNN-style (Undir-Node)   |       0 | illicit |       0.785 |   nan     |    nan     |      0.215 |
| AF         | CAGNN-style (Undir-Node)   |       0 | licit   |       0.691 |   nan     |    nan     |      0.309 |
| AF         | CAGNN-style (Undir-Node)   |       1 | illicit |       0.546 |   nan     |    nan     |      0.454 |
| AF         | CAGNN-style (Undir-Node)   |       1 | licit   |       0.474 |   nan     |    nan     |      0.526 |
| AF         | Dir-Global                 |       0 | illicit |       0.431 |     0.262 |      0.308 |    nan     |
| AF         | Dir-Global                 |       0 | licit   |       0.431 |     0.262 |      0.308 |    nan     |
| AF         | Dir-Global                 |       1 | illicit |       0.374 |     0.285 |      0.341 |    nan     |
| AF         | Dir-Global                 |       1 | licit   |       0.374 |     0.285 |      0.341 |    nan     |
| AF         | NADir (Dir-Node, proposed) |       0 | illicit |       0.51  |     0.207 |      0.284 |    nan     |
| AF         | NADir (Dir-Node, proposed) |       0 | licit   |       0.384 |     0.386 |      0.23  |    nan     |
| AF         | NADir (Dir-Node, proposed) |       1 | illicit |       0.414 |     0.18  |      0.406 |    nan     |
| AF         | NADir (Dir-Node, proposed) |       1 | licit   |       0.366 |     0.225 |      0.409 |    nan     |
| AF         | Undir-Global               |       0 | illicit |       0.661 |   nan     |    nan     |      0.339 |
| AF         | Undir-Global               |       0 | licit   |       0.661 |   nan     |    nan     |      0.339 |
| AF         | Undir-Global               |       1 | illicit |       0.539 |   nan     |    nan     |      0.461 |
| AF         | Undir-Global               |       1 | licit   |       0.539 |   nan     |    nan     |      0.461 |
| LF         | CAGNN-style (Undir-Node)   |       0 | illicit |       0.749 |   nan     |    nan     |      0.251 |
| LF         | CAGNN-style (Undir-Node)   |       0 | licit   |       0.557 |   nan     |    nan     |      0.443 |
| LF         | CAGNN-style (Undir-Node)   |       1 | illicit |       0.64  |   nan     |    nan     |      0.36  |
| LF         | CAGNN-style (Undir-Node)   |       1 | licit   |       0.527 |   nan     |    nan     |      0.473 |
| LF         | Dir-Global                 |       0 | illicit |       0.44  |     0.245 |      0.316 |    nan     |
| LF         | Dir-Global                 |       0 | licit   |       0.439 |     0.245 |      0.316 |    nan     |
| LF         | Dir-Global                 |       1 | illicit |       0.373 |     0.283 |      0.345 |    nan     |
| LF         | Dir-Global                 |       1 | licit   |       0.373 |     0.283 |      0.345 |    nan     |
| LF         | NADir (Dir-Node, proposed) |       0 | illicit |       0.568 |     0.193 |      0.239 |    nan     |
| LF         | NADir (Dir-Node, proposed) |       0 | licit   |       0.373 |     0.354 |      0.273 |    nan     |
| LF         | NADir (Dir-Node, proposed) |       1 | illicit |       0.437 |     0.261 |      0.302 |    nan     |
| LF         | NADir (Dir-Node, proposed) |       1 | licit   |       0.268 |     0.422 |      0.311 |    nan     |
| LF         | Undir-Global               |       0 | illicit |       0.667 |   nan     |    nan     |      0.333 |
| LF         | Undir-Global               |       0 | licit   |       0.667 |   nan     |    nan     |      0.333 |
| LF         | Undir-Global               |       1 | illicit |       0.555 |   nan     |    nan     |      0.445 |
| LF         | Undir-Global               |       1 | licit   |       0.555 |   nan     |    nan     |      0.445 |
