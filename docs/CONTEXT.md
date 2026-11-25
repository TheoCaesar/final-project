# Project context — read this first

Start-of-session briefing for anyone (human or AI assistant) resuming work on this repo.
Keep it current: update **Status** and **Open questions** at the end of every session. Last updated: 2026-11-25 (S01).

## 1. What this project is

MSc thesis, Department of Computer Science, University of Ghana (Data Intelligence & Swarm Analytics Lab).
Supervisor: Prof. Justice Kwame Appati.
Title: **Heterophily-Aware GNNs for Money Laundering Detection**: node-adaptive directional aggregation on AML
transaction graphs.

**Research question:** Does heterophily in AML transaction graphs depend on edge direction, i.e. do illicit nodes'
incoming and outgoing neighbourhoods differ? If so, does letting each node weight its own features, its in-neighbours
and its out-neighbours separately (NADir) improve illicit-node detection over undirected and globally-weighted
directional GNNs?

**Guiding principle (from the research brief):** optimise for a rigorously explained finding, not for a novel model.
Do not overclaim. If evidence contradicts the hypothesis, report it and pivot.

## 2. Status (end of S01, 2026-11-25)

| Stage | State | Key result |
|---|---|---|
| 1 Literature verification | partial | Key refs: CAGNN (Chen et al. 2024), DirGNN (Rossi et al. 2023), GCPAL / AMLworld, Weber et al. 2019. Multi-GNN (Egressy et al. 2024) still to be added. |
| 2 Direction analysis, Elliptic | **done** | Illicit h_in 0.663 vs h_out 0.521; edge-flip test p < 0.001; gap holds in train and test periods. **Hypothesis supported.** |
| 3 Baselines, Elliptic | **done** | LF: GNNs ≈ MLP (~0.74 PR-AUC); GCN worse (0.656). AF: RF best (0.764), neural models unstable. |
| 4 NADir, Elliptic | **done** | **Does not beat baselines** (LF 0.738 vs MLP 0.739 / CAGNN-style 0.749; AF 0.348). Gates trust neighbourhoods less for illicit nodes. |
| AMLworld | not started | Blocked on task definition (see §6). |

Current framing: *direction-dependent heterophily is real on Elliptic but not exploitable under its temporal shift.*
Do not claim NADir improves AML detection on present evidence.

## 3. Repo map

```
data/elliptic/            Kaggle zips (not in git): elliptic_txs_{classes,edgelist,features}.csv.zip
src/data.py               loader → directed PyG graph; temporal split; standardisation; synthetic test data
src/homophily.py          Stage 2: in/out homophily, transition matrices, edge-flip permutation test
src/models.py             MLP/GCN/GAT/SAGE/DirGNN + NAMix layer (2×2 ablation; NADir = directed + node gate)
src/train.py              training/eval loop, resumable (skips runs already in runs.csv)
src/summarize.py          runs.csv → summary.md (mean ± sd)
src/figures.py            thesis figures (PNG + PDF)
results/                  homophily.{md,json}, runs.csv, gates.csv, summary.md, train.log, fig_*.{png,pdf}
docs/CONTEXT.md           this file: briefing
docs/REPRO_LOG.md         detailed, timestamped log of every step and decision (append-only)
docs/SESSIONS.md          one short entry per session; session ID ↔ commit tag
```

## 4. Fixed experimental setup (change only with a REPRO_LOG entry)

- Edge direction: `txId1 → txId2` (funds flow). In-neighbours = sources, out-neighbours = destinations.
- Labels: illicit 1, licit 0, unknown −1 (unknown kept in graph, excluded from loss and metrics).
- Split: train steps 1–29, val 30–34, test 35–49. Elliptic has no cross-step edges, so training runs on exact
  time-window subgraphs.
- Feature sets: LF (93 local), AF (165 = local + 72 aggregated); z-scored on steps 1–34.
- Models: hidden 64, 2 layers, LayerNorm, dropout 0.5, Adam lr 0.01, wd 5e-4, ≤200 epochs, patience 25 on val PR-AUC,
  class-weighted CE. Seeds 0, 1, 2. `--threads 2`.
- Metrics: PR-AUC (primary), ROC-AUC, illicit F1/precision/recall at the val-tuned threshold and at 0.5,
  F1 before/after the dark-market shutdown (step 43).

## 5. Working conventions

- **Every session:** read this file → skim the latest entries of `REPRO_LOG.md` and `SESSIONS.md` → work →
  append dated sections to `REPRO_LOG.md` (`## YYYY-MM-DD HH:MM UTC — title`) → add the session entry to `SESSIONS.md`
  → update §2 and §6 here → commit with subject prefix `[Snn]`.
- REPRO_LOG is append-only; supersede decisions with a new entry, never edit old ones.
- New result folders for replications (`--out results_<tag>`), so `results/` stays the reference run.
- Environment: Windows, Python 3.14.7, `.venv` (`python -m venv .venv`, `.venv\Scripts\activate`,
  `pip install -r requirements.txt`). Reference results were produced on Linux, Python 3.13.18, CPU.
- Known caveat: AF neural results change with CPU thread count (same seed: 0.51 vs 0.29 PR-AUC).

## 6. Open questions / next steps

1. **AMLworld task definition:** labels are on transactions (edges). Options: edge classification, as in Multi-GNN,
   or account labels derived from transactions (rule to be justified). Decide before writing a loader.
2. Re-run Stage 2 on AMLworld HI-Small: is the in/out asymmetry present where graph structure matters more?
3. Add Multi-GNN (Egressy et al., AAAI 2024) to the literature comparison and, if feasible, as a baseline.
4. Strengthen Elliptic evidence if time allows: 5 seeds, small hyperparameter search, DirGNN α ∈ {0, 0.5, 1}.
5. Thesis framing: position the contribution as the directional-heterophily analysis plus a controlled ablation,
   not as a new SOTA model.

## 7. Quick commands

```
python src\homophily.py --raw data\elliptic --out results --perms 1000
python src\train.py --raw data\elliptic --out results --seeds 0,1,2 --features LF,AF --epochs 200 --patience 25 --threads 2
python src\summarize.py --out results
python src\figures.py --out results
```
