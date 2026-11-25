# Direction-aware homophily analysis (Elliptic)

Nodes 203,769; edges 234,355; edges with both ends labelled 36,624 (15.6%). Cross-time-step edges: 0.

Edge homophily 0.954; adjusted homophily (undirected) 0.516.

## Where does money go? P(dst class | src class)

| src \ dst | licit | illicit |
|---|---|---|
| licit | 0.977 | 0.023 |
| illicit | 0.478 | 0.522 |

## Where does money come from? P(src class | dst class)

| dst \ src | licit | illicit |
|---|---|---|
| licit | 0.974 | 0.026 |
| illicit | 0.439 | 0.561 |

## Per-class directional homophily (all time steps)

| metric | illicit | licit |
|---|---|---|
| n_nodes | 4,545 | 42,019 |
| mean_in_degree | 1.270 | 1.909 |
| mean_out_degree | 0.742 | 1.186 |
| share_no_in_edges | 0.265 | 0.332 |
| share_no_out_edges | 0.326 | 0.245 |
| coverage_in (>=1 labelled in-nbr) | 0.300 | 0.478 |
| coverage_out (>=1 labelled out-nbr) | 0.391 | 0.604 |
| h_in (mean over covered nodes) | 0.663 | 0.993 |
| h_out (mean over covered nodes) | 0.521 | 0.979 |

## Edge-flip permutation test (H0: direction carries no class information)

| class | h_out - h_in (observed) | null mean ± sd | p (two-sided) |
|---|---|---|---|
| illicit | -0.142 | +0.001 ± 0.014 | 0.0000 |
| licit | -0.014 | -0.000 ± 0.001 | 0.0000 |

## Train period (1-34) vs test period (35-49)

| metric | illicit 1-34 | illicit 35-49 | licit 1-34 | licit 35-49 |
|---|---|---|---|---|
| h_in (mean over covered nodes) | 0.704 | 0.480 | 0.991 | 0.996 |
| h_out (mean over covered nodes) | 0.568 | 0.356 | 0.974 | 0.987 |
| mean_in_degree | 1.263 | 1.292 | 1.825 | 2.052 |
| mean_out_degree | 0.767 | 0.661 | 1.146 | 1.253 |
