# CLAUDE.md

MSc thesis repo: heterophily-aware, directional GNNs for money-laundering detection (Elliptic; AMLworld planned).

## Start of every session
1. Read `docs/CONTEXT.md` (briefing: question, status, setup, open questions).
2. Skim the last few entries of `docs/REPRO_LOG.md` and `docs/SESSIONS.md`.
3. Find the next session ID in `docs/SESSIONS.md` (`S02`, `S03`, …) and use it for this session.

## While working
- Log every step and decision in `docs/REPRO_LOG.md` as a new section `## YYYY-MM-DD HH:MM UTC — title`.
  Append-only: never edit old entries; supersede them with a new one.
- Any change to the fixed setup (`docs/CONTEXT.md` §4: split, features, hyperparameters, seeds, metrics) needs a
  REPRO_LOG entry explaining why.
- Replications go to a new folder (`--out results_<tag>`); `results/` is the reference run.
- Do not commit `data/`, `.venv/`, `*.pt` (see `.gitignore`).

## End of every session
- Add the session entry to `docs/SESSIONS.md` (Done / Files / Decisions / Open / Environment) and its index row.
- Update `docs/CONTEXT.md` §2 Status and §6 Open questions, and its "Last updated" line.
- Propose a commit message whose subject starts with `[Snn]`; after the user commits, record the hash in `SESSIONS.md`.

## Research standards (from the research brief)
- Optimise for a rigorously explained finding, not a novel model. Report negative results plainly.
- Be critical: flag unsupported claims, weak baselines, leakage, overclaiming, and novelty already covered by
  prior work (DirGNN, CAGNN, Multi-GNN).
- Current evidence: direction-dependent heterophily exists on Elliptic, but NADir does **not** beat baselines.
  Do not write otherwise without new results.

## Environment
Windows, Python 3.14.7, `.venv`: `python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`.
Reference results: Linux, Python 3.13.18, CPU, `--threads 2`. Commands: see `docs/CONTEXT.md` §7.

## Layout
`src/` code · `results/` outputs · `docs/` CONTEXT, REPRO_LOG, SESSIONS · `data/elliptic/` Kaggle zips (not in git)
