# Session log

One entry per working session, newest at the bottom. Each session has an ID (`S01`, `S02`, …) that also goes in
the subject line of the commit(s) made at the end of that session, so sessions and commits can be matched:

```
git log --oneline --grep "\[S01\]"      # commits belonging to session S01
```

After committing, fill in the commit hash (`git log -1 --format=%h`) in the index and in the session entry.
Detailed, timestamped reasoning for every step lives in [`REPRO_LOG.md`](REPRO_LOG.md); this file is the short tally.

## Index

| Session | Date (UTC) | Focus | Commit(s) |
|---|---|---|---|
| S01 | 2026-11-25 | Pipeline, direction analysis, baselines + NADir on Elliptic | _fill after commit_ |

---

## S01 — 2026-11-25, 10:53–14:10 UTC

**Goal:** Test whether heterophily in AML transaction graphs depends on edge direction, and whether a node-adaptive
directional GNN (NADir) exploits it. Time-boxed to a few hours, so Elliptic only.

**Done**
- Set up the repo: `src/` pipeline, `requirements.txt` (pinned), `.gitignore` (data, caches, `.venv/` excluded).
- Loaded Elliptic from the three Kaggle zips in `data/elliptic/`; counts verified against the published statistics
  (203,769 nodes, 234,355 edges, 4,545 illicit, 42,019 licit).
- Stage 2, direction-aware homophily (`src/homophily.py`): illicit h_in 0.663 vs h_out 0.521; edge-flip permutation
  test p < 0.001. Gap holds in both the train and test periods.
- Stage 3/4, models (`src/models.py`, `src/train.py`): RF, MLP, GCN, GAT, SAGE, DirGNN and a 2×2 ablation
  (directed/undirected × global/node gate), with NADir = directed + node gate. Temporal split 1–29 / 30–34 / 35–49;
  2 feature sets × 10 models × 3 seeds = 60 runs.
- Results: NADir does not beat the baselines (LF PR-AUC 0.738 vs MLP 0.739, CAGNN-style 0.749; on AF it is among the
  worst, while RF is best at 0.764). Gates down-weight neighbourhoods for illicit nodes.
- Figures (`src/figures.py`): `fig_direction_homophily`, `fig_model_prauc` (PNG + PDF).
- Reproducibility log started (`docs/REPRO_LOG.md`, 14 dated entries), including the deviations
  (3 seeds instead of 5, no tuning) and the thread-count non-determinism on AF.

**Files added**
`src/{data,homophily,models,train,summarize,figures}.py`, `requirements.txt`, `.gitignore`,
`docs/REPRO_LOG.md`, `docs/SESSIONS.md`, `docs/CONTEXT.md`, `CLAUDE.md`,
`results/{homophily.md,homophily.json,runs.csv,gates.csv,summary.md,train.log}`,
`results/fig_{direction_homophily,model_prauc}.{png,pdf}`.
Not committed: `data/` (raw Kaggle zips; get them from Kaggle `ellipticco/elliptic-data-set`).

**Decisions**
- Elliptic only; AMLworld deferred until the task definition is settled (transaction vs account labels).
- PR-AUC as the primary metric; F1 at the val-tuned threshold and at 0.5 both reported.
- Clipping heavy-tailed features was tried as a sensitivity check only, not adopted.

**Open / next session**
- Decide the AMLworld task (edge classification vs derived account labels), then rerun Stages 2–4 there.
- Add Multi-GNN (Egressy et al., 2024) as a baseline/literature comparison.
- If time allows: 5 seeds, small hyperparameter search, DirGNN α ∈ {0, 0.5, 1}.

**Environment:** results produced on Linux, Python 3.13.18, CPU (2 threads). Local machine: Windows, Python 3.14.7, `.venv`.

**Commit:** _fill after commit_
