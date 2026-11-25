# Reproducibility log — Heterophily-aware directional GNNs for AML

Thesis: *Heterophily-Aware GNNs for Money Laundering Detection* (MSc, University of Ghana).
Convention: one section per step, headed with the UTC date and time it was done. Append, never rewrite;
if a decision changes, add a new section that supersedes the old one and say so.

---

## 2026-11-25 10:53 UTC — Plan and scope decision

- **Decision:** Do the data analysis before the novel model. The motivating claim (heterophily in AML graphs
  depends on edge direction) is a hypothesis, not an established fact, so it is tested first (Stage 2), then
  baselines (Stage 3), then the proposed model (Stage 4). This order comes from the research brief (`postchatpormpt.pdf`, §12).
- **Time constraint (stated 10:56):** results are needed within hours, so the scope is cut to **Elliptic only**.
  AMLworld (HI-Small) is deferred: its labels are on transactions (edges), not accounts, so it needs its own
  task definition (edge classification, or account labels derived from transactions), and that decision must not be rushed.
- **Known novelty risk to address in the thesis:** DirGNN (Rossi et al., 2023) already uses separate in/out
  aggregation with a global α; Multi-GNN (Egressy et al., AAAI 2024) shows reverse message passing helps on
  AMLworld. The contribution tested here is narrower: whether a **per-node** (rather than global) mixing of
  self / in / out messages helps.

## 2026-11-25 10:57 UTC — Compute environment

- The user's machine (linked VM) has 2 cores, 3 GB RAM and no PyTorch, so experiments run in a cloud workspace:
  Linux x86_64, 2 CPU cores, 7 GB RAM, **no GPU**, Python 3.13.18.
- Package versions pinned in `requirements.txt` (torch 2.15.0 CPU, torch-geometric 2.8.1, scikit-learn 1.9.1, …).
- Every model is full-batch on CPU; all runs are seeded (`torch.manual_seed`, `np.random.seed`), seeds 0–4.
  CPU float ops are not bit-identical across machines, so expect small differences on re-run.

## 2026-11-25 11:00 UTC — Data acquisition

- Source: Elliptic Bitcoin dataset (Weber et al., 2019), Kaggle `ellipticco/elliptic-data-set`.
  Three files: `elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`.
- PyG's mirror (data.pyg.org), Kaggle and Hugging Face are not reachable from the workspace, and GitHub mirrors
  store the files in Git LFS, which the workspace can't download. The user downloads the files from Kaggle into
  `data/elliptic/`. The loader (`src/data.py`) accepts a directory or zip(s) containing the three CSVs.
- Expected sizes (to verify on load): 203,769 nodes; 234,355 directed edges; 4,545 illicit / 42,019 licit /
  157,205 unknown; 49 time steps; 165 features (93 local + 72 one-hop aggregated) after dropping txId and time step.

## 2026-11-25 11:01 UTC — Graph construction and split (design decisions)

- **Direction:** edgelist row `(txId1, txId2)` = BTC output of txId1 spent as input of txId2, so the edge is `txId1 → txId2`.
  In-neighbours of v = predecessors (where the funds came from); out-neighbours = successors (where they went).
- **Labels:** illicit = 1, licit = 0, unknown = −1. Unknown nodes stay in the graph (they pass messages) but are never
  used for loss or metrics.
- **Split (temporal, no random splits):** train steps 1–29, validation 30–34 (early stopping + threshold choice),
  test 35–49. This follows the standard train 1–34 / test 35–49 split of Weber et al. (2019), with the
  last five training steps held out for validation.
  Elliptic has no edges between time steps, so test nodes are effectively unseen subgraphs (inductive).
- **Features:** two settings: AF = all 165 features; LF = 93 local features only. The AF aggregated features already
  summarise one-hop neighbourhoods, so graph gains are expected to be smaller on AF; LF is the cleaner test of message passing.
  Standardised with mean/sd from steps 1–34 only.
- **Imbalance:** weighted cross-entropy, illicit weight = #licit/#illicit in train.
- **Metrics:** test PR-AUC (primary; threshold-free and suited to rare positives), ROC-AUC, illicit-class
  F1/precision/recall at a threshold tuned on validation F1 and at 0.5, plus F1 before vs after the dark-market
  shutdown (test steps 35–42 vs 43–49), where Weber et al. report all models degrade.

## 2026-11-25 11:02 UTC — Stage 2 method: direction-aware homophily (`src/homophily.py`)

Computed on edges whose two endpoints are labelled (coverage reported):
1. Edge homophily and adjusted homophily (Platonov et al., 2023) on the undirected graph: the reference point.
2. Class-transition matrices P(dst class | src class) and P(src class | dst class): where illicit money goes vs where it comes from.
3. Per-class node-level **h_in** (share of labelled in-neighbours with the same label) and **h_out**, plus degree
   asymmetry and the share of nodes with no in-/out-edges.
4. **Edge-flip permutation test**: H0 = direction carries no class information. Each edge is reversed with p = 0.5,
   h_out − h_in is recomputed (1000 permutations), and a two-sided p-value is reported.
5. The same statistics for steps 1–34 vs 35–49 (distribution shift).

**Sanity check on synthetic data** (random edges, same schema; `make_synthetic`): flip-test p = 0.40 (illicit) and
0.54 (licit), so no false direction effect is detected. Pipeline verified end to end.

## 2026-11-25 11:03 UTC — Stage 3/4 method: models (`src/models.py`, `src/train.py`)

Shared: linear encoder → 2 message-passing layers (hidden 64, LayerNorm, ReLU, dropout 0.5) → linear classifier.
Adam, lr 0.01, weight decay 5e-4, ≤300 epochs, early stopping on validation PR-AUC (patience 40). Seeds 0–4.
No hyperparameter search was done, because of the time constraint; all neural models share the same settings (stated as a limitation).

Baselines: RF (50 trees, max_features 50, as in Weber et al.), MLP, GCN, GAT (4 heads), GraphSAGE (undirected),
DirGNN (Dir-SAGE, Rossi et al. 2023, α = 0.5 fixed).

**Proposed layer family `NAMix`:** messages m_self = W_s h_v, m_in = mean_{u→v} W_in h_u, m_out = mean_{v→w} W_out h_w,
combined as h'_v = Σ_k g_{v,k} m_k with gates g_v = softmax(MLP([m_self ‖ m_in ‖ m_out ‖ log(1+deg_in) ‖ log(1+deg_out)])).
The degree inputs let the gate see when a node has no in- or out-neighbours.
**2×2 factorial ablation**, all with identical code:

| | global gate (one per layer) | node-adaptive gate |
|---|---|---|
| undirected neighbours | Undir-Global | CAGNN-style (Chen et al., 2024) |
| directed in/out | Dir-Global (≈ DirGNN with learned α) | **NADir (proposed)** |

Gate weights on test nodes are saved by class (`results/gates.csv`) for interpretability: do illicit nodes rely on in vs out differently?

**Commands:**
```
pip install -r requirements.txt
python src/homophily.py --raw data/elliptic --out results --perms 1000
python src/train.py     --raw data/elliptic --out results --seeds 0,1,2,3,4 --features AF,LF
python src/summarize.py --out results
```

## 2026-11-25 11:10 UTC — Data received and verified

- Kaggle per-file zips placed by the user in `data/elliptic/`. SHA-256:
  - `elliptic_txs_classes.csv.zip`  4ca957f0ceffd5dd164e255c7d5ad9ee69a6fa64ae1dd94d6f113e5ebf3b07ba
  - `elliptic_txs_edgelist.csv.zip` a2f9f6b67a39da2d8cf87fe77b9db89571ba6d880e5dd5b5991dc45c80fa34ec
  - `elliptic_txs_features.csv.zip` d33d62159e64b5e889f1a7ea880227c612775b58d409598855e0c4400fa52b3e
- Loaded counts match the published statistics: 203,769 nodes, 234,355 edges, 4,545 illicit, 42,019 licit,
  0 edges crossing time steps.

## 2026-11-25 11:12 UTC — Stage 2 RESULT: heterophily in Elliptic is direction-dependent

Full tables: `results/homophily.md` / `results/homophily.json` (`python src/homophily.py --raw data/elliptic --perms 1000`, ~17 s).

- Undirected view: edge homophily 0.954, but adjusted homophily only 0.516. The raw number is inflated by
  the licit majority; the graph is far less homophilous once class imbalance is corrected for.
- Licit nodes are homophilous in both directions (h_in 0.993, h_out 0.979).
- **Illicit nodes are not:** h_in = 0.663 (two-thirds of a labelled illicit transaction's funding sources are illicit),
  h_out = 0.521 (only about half of where its money goes is illicit). Difference −0.142.
- **Edge-flip permutation test:** null mean +0.001 ± 0.014; observed −0.142 is about 10 null SDs; 0/1000 permutations
  as extreme (p < 0.001). Licit: −0.014, also p < 0.001, but practically small.
- Edge-level transition matrix tells the same story, with a smaller gap: P(dst illicit | src illicit) = 0.522 vs
  P(src illicit | dst illicit) = 0.561. The gap is larger per node (0.14) than per edge (0.04) because the node-level mean
  weights every illicit node equally while the edge level is dominated by high-degree nodes. Report both.
- Degree asymmetry: illicit nodes have fewer out-edges (mean out-degree 0.74 vs 1.19 for licit) and are more often
  sinks (32.6% have no out-edge vs 24.5% for licit).
- **Interpretation (hypothesis-consistent, not proof of mechanism):** illicit funds tend to arrive from illicit
  sources and leave into licit transactions, consistent with layering/integration. A symmetric (undirected)
  aggregator mixes a relatively homophilous incoming neighbourhood with a heterophilous outgoing one.
- **Distribution shift:** illicit homophily falls sharply in the test period (h_in 0.70 → 0.48, h_out 0.57 → 0.36
  for steps 1–34 vs 35–49), but the in > out gap persists (0.14 and 0.12). Neighbour labels are less informative
  at test time, so graph models are expected to lose ground to feature-only models there.
- **Caveats:** label-based statistics use the 15.6% of edges with both ends labelled; only 30% (in) / 39% (out) of
  illicit nodes have any labelled neighbour in that direction. A permutation test also cannot rule out that
  labelling practice itself (how Elliptic assigned labels) contributes to the asymmetry. State this in the thesis.

**Consequence for the thesis:** the precondition for a directional model is supported on Elliptic. Stage 3/4 now
tests whether exploiting it improves detection.

## 2026-11-25 11:16 UTC — Compute adjustment (deviation from 11:03 plan)

- Measured per-epoch cost on full graph, 2 CPUs: 1.0 s (MLP) to 3.5 s (NADir). Too slow for 5 seeds × 2 feature sets
  in the available time.
- Change 1 (exact, no effect on results): since Elliptic has no cross-time-step edges, training uses the induced
  subgraph of steps 1–29, validation the subgraph of 30–34, test the subgraph of 35–49. Each node's messages are
  identical to the full-graph computation.
- Change 2 (reduces statistical power): **3 seeds (0, 1, 2)** instead of 5; ≤200 epochs, patience 25.
- Command actually run:
  `python src/train.py --raw data/elliptic --out results --seeds 0,1,2 --features LF,AF --epochs 200 --patience 25`

## 2026-11-25 12:05 UTC — Stage 3/4 RESULT (local features, LF): proposed model does not beat baselines

Full table: `results/summary.md`; per-run rows: `results/runs.csv`. Test steps 35–49, 3 seeds.

| Model | PR-AUC | F1 @ val-threshold |
|---|---|---|
| RF | 0.741 ± 0.005 | 0.709 ± 0.058 |
| MLP | 0.739 ± 0.017 | 0.755 ± 0.022 |
| GCN | 0.656 ± 0.015 | 0.595 ± 0.062 |
| GAT | 0.745 ± 0.015 | 0.747 ± 0.008 |
| SAGE | 0.723 ± 0.023 | 0.701 ± 0.010 |
| DirGNN (α=0.5) | 0.716 ± 0.008 | 0.673 ± 0.041 |
| Undir-Global | 0.744 ± 0.005 | 0.716 ± 0.002 |
| CAGNN-style (Undir-Node) | 0.749 ± 0.010 | 0.738 ± 0.011 |
| Dir-Global | 0.728 ± 0.021 | 0.710 ± 0.024 |
| NADir (proposed) | 0.738 ± 0.044 | 0.712 ± 0.047 |

- No graph model is meaningfully better than the feature-only MLP; with 3 seeds, differences of ±0.01–0.02 are within noise.
  Plain GCN is clearly worse (0.656), consistent with Weber et al. (2019), where GCN trailed RF.
- **Direction did not help:** Dir-Global ≤ Undir-Global, and NADir ≈ CAGNN-style but with ~4× the seed variance.
- Every model has F1 ≈ 0 after the dark-market shutdown (steps 43–49), as reported by Weber et al.
- F1 at threshold 0.5 is low for neural models (0.27–0.51) because the class-weighted loss inflates illicit
  probabilities; the val-tuned threshold is the fair comparison. Report both.
- **Gate interpretability (NADir, layer 1, test nodes, mean over seeds):** illicit nodes get gate_self 0.57 / in 0.19 / out 0.24;
  licit nodes 0.37 / 0.35 / 0.27. The node gate learns to trust the neighbourhood less for illicit nodes, consistent with
  their lower homophily. It does **not** simply favour the more homophilous in-direction for illicit nodes
  (gate_in < gate_out), so the gate is not reproducing the Stage 2 asymmetry directly.
- **Honest reading:** Stage 2 shows direction-dependent heterophily exists on Elliptic, but on this dataset and split it
  does not translate into better detection. Likely reasons, all measured above: (i) only 15.6% of edges have both ends
  labelled and many illicit nodes have no labelled neighbour; (ii) illicit homophily drops sharply in the test period,
  so neighbourhood patterns learned on steps 1–29 go stale; (iii) the local features are already strong (MLP ≈ RF).

## 2026-11-25 12:15 UTC — Diagnostic: all-features (AF) setting and non-determinism

- Neural models on AF are much worse than on LF (MLP PR-AUC ~0.50 vs 0.74; GCN ~0.42). Weber et al. also report
  MLP worse on AF than LF (F1 0.628 vs 0.694), so the direction of the effect is expected; the size is larger here.
- Checked for a preprocessing bug: no feature has train sd < 0.23, so standardisation is well-posed. The features are very
  heavy-tailed (test |z| up to ~1,200 in local, ~190 in aggregated features). Clipping z to ±10 did not fix AF
  (MLP seed 0: 0.29 → 0.33 PR-AUC) but slightly improved LF (F1 0.73 → 0.77). Not adopted for the main runs, so the
  pre-registered setup stays unchanged; noted as a sensitivity check.
- **Reproducibility warning:** the same seed gave AF-MLP PR-AUC 0.507 with 2 CPU threads and 0.289 with 1 thread.
  Floating-point reduction order changes with thread count; on AF this small difference changes the early-stopping
  epoch and the final model. All main results use `--threads 2`. AF results are therefore unstable and should be
  reported with that caveat; LF results varied far less (MLP seed 0: 0.719 vs 0.719–0.726).

## 2026-11-25 13:20 UTC — Stage 3/4 RESULT (all features, AF) and overall conclusion

All 60 runs finished (10 models × 2 feature sets × 3 seeds; no duplicates). Test PR-AUC, AF:

| Model | PR-AUC | F1 @ val-thr | F1 @ 0.5 |
|---|---|---|---|
| RF | **0.764 ± 0.003** | 0.480 ± 0.085 | **0.703 ± 0.031** |
| MLP | 0.504 ± 0.020 | 0.508 ± 0.012 | 0.383 ± 0.012 |
| GCN | 0.418 ± 0.075 | 0.526 ± 0.036 | 0.418 ± 0.053 |
| GAT | 0.429 ± 0.067 | 0.488 ± 0.059 | 0.485 ± 0.037 |
| SAGE | 0.431 ± 0.032 | 0.526 ± 0.015 | 0.483 ± 0.038 |
| DirGNN (α=0.5) | 0.498 ± 0.038 | 0.505 ± 0.009 | 0.497 ± 0.014 |
| Undir-Global | 0.510 ± 0.063 | 0.548 ± 0.011 | 0.503 ± 0.043 |
| CAGNN-style (Undir-Node) | 0.311 ± 0.021 | 0.476 ± 0.052 | 0.428 ± 0.089 |
| Dir-Global | 0.435 ± 0.038 | 0.505 ± 0.012 | 0.479 ± 0.032 |
| NADir (proposed) | 0.348 ± 0.036 | 0.480 ± 0.028 | 0.467 ± 0.009 |

- RF on AF is the best model overall (PR-AUC 0.764), matching the literature finding that tree ensembles are
  hard to beat on Elliptic. Its val-tuned threshold transfers badly (F1 0.48 vs 0.70 at 0.5).
- All neural models degrade on AF, and the **node-gated models degrade most** (CAGNN-style 0.31, NADir 0.35). The extra
  gate capacity appears to overfit the training-period structure carried in the 72 neighbourhood-aggregated features. Given the
  thread-count instability (12:15 entry), treat AF neural rankings as low-confidence.

**Overall conclusion on Elliptic (as of this entry):**
1. *Supported:* heterophily of illicit transactions is direction-dependent (h_in 0.66 vs h_out 0.52; p < 0.001), and
   the asymmetry persists in both time periods.
2. *Not supported:* exploiting this asymmetry with node-adaptive directional aggregation does not improve detection.
   NADir ≈ baselines on LF (0.738 vs MLP 0.739, CAGNN-style 0.749) and is among the worst on AF. Direction alone
   (Dir-Global vs Undir-Global) also does not help on either feature set.
3. *Explanatory evidence:* sparse labelled edges, sharp drop of illicit homophily in the test period, strong node features.

**Implication for the thesis:** this is a negative-but-informative result on Elliptic. A defensible framing is
"directional heterophily is real but, under Elliptic's temporal shift, not exploitable", with AMLworld (where graph
structure is known to matter more; Egressy et al. 2024) as the test of whether the hypothesis holds where structure is
informative. Do not claim NADir improves AML detection on the current evidence.

**Limitations to state:** 3 seeds; no hyperparameter tuning (shared defaults); α fixed at 0.5 for DirGNN; single dataset;
CPU float non-determinism across thread counts.

## 2026-11-25 13:25 UTC — Artefacts

`results/homophily.{md,json}`, `results/runs.csv` (one row per run), `results/gates.csv`, `results/summary.md`,
`results/train.log`, `results/fig_direction_homophily.{png,pdf}`, `results/fig_model_prauc.{png,pdf}`.
Regenerate summaries/figures: `python src/summarize.py --out results && python src/figures.py --out results`.

## 2026-11-25 12:55 UTC — Repository layout and local environment

- This log now lives at `docs/REPRO_LOG.md` (moved by the user). All later entries are appended here.
- **Data:** only the three Kaggle Elliptic files are needed, kept zipped in `data/elliptic/`
  (`elliptic_txs_{classes,edgelist,features}.csv.zip`; the loader reads the zips directly). No other dataset is used
  in the Elliptic experiments. If AMLworld is added later it goes in its own folder (`data/amlworld/`), not in `data/elliptic/`.
- **Local Python:** the user's machine runs Python 3.14.7 (Windows); the reported results were produced on Python 3.13.18
  (Linux, CPU). Every pinned package in `requirements.txt` has a cp314 Windows wheel (checked on PyPI), so the same versions
  install. Different OS, Python and thread count can shift results slightly (see 12:15 entry); re-runs on Windows should
  match LF numbers closely, while AF neural numbers may differ more.
- **Virtual environment (recommended, kept out of git):**
  ```
  cd C:\Users\nexa\Desktop\school\final-project
  python -m venv .venv
  .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  python src\homophily.py --raw data\elliptic --out results --perms 1000
  python src\train.py --raw data\elliptic --out results --seeds 0,1,2 --features LF,AF --epochs 200 --patience 25 --threads 2
  python src\summarize.py --out results
  python src\figures.py --out results
  ```
  `pip install torch` on Windows installs the CPU build, which is what the results used. Re-running `train.py` into an existing
  `results/` folder skips runs already in `runs.csv`; use a new `--out` folder for a fresh replication.

## 2026-11-25 14:10 UTC — Session tracking

- Added `docs/SESSIONS.md`: one short entry per working session with an ID (`S01`, …) that also appears in the commit
  subject, so `git log --grep "\[S01\]"` maps a session to its commits. This session is S01 (the repo's first commit).

## 2026-11-25 14:15 UTC — Context files for resuming work

- Added `docs/CONTEXT.md` (briefing: research question, status table, repo map, fixed setup, conventions, open questions,
  commands) and `CLAUDE.md` at the repo root (start/during/end-of-session rules; loaded automatically by Claude Code).
- A copy of CONTEXT.md is also stored in the claude.ai Project "AML_thesis" as `claude/CONTEXT.md`.
