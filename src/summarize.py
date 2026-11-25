"""Aggregate results/runs.csv and gates.csv into results/summary.md (mean ± sd over seeds)."""
import argparse, os
import pandas as pd

COLS = ["test_pr_auc", "test_roc_auc", "test_f1@val", "test_precision@val", "test_recall@val",
        "test_f1@0.5", "test_f1@val_pre43", "test_f1@val_post43"]


def fmt(g, c):
    return f"{g[c].mean():.3f} ± {g[c].std(ddof=1) if len(g) > 1 else 0:.3f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
    r = pd.read_csv(os.path.join(a.out, "runs.csv"))
    order = list(dict.fromkeys(r["model"]))
    L = ["# Results — Elliptic, temporal split (train 1–29, val 30–34, test 35–49)\n",
         "Mean ± sd over seeds. Illicit = positive class. `@val` = threshold chosen on validation F1. "
         "pre43/post43 = test steps before/after the dark-market shutdown.\n"]
    for feat in r["features"].unique():
        sub = r[r.features == feat]
        L += [f"\n## Features: {feat} ({'all 165' if feat == 'AF' else 'local 93 only'})\n",
              "| model | n | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 2) + "|"]
        for m in order:
            g = sub[sub.model == m]
            if len(g):
                L.append(f"| {m} | {len(g)} | " + " | ".join(fmt(g, c) for c in COLS) + " |")
    gp = os.path.join(a.out, "gates.csv")
    if os.path.exists(gp):
        gt = pd.read_csv(gp)
        agg = gt.groupby(["features", "model", "layer", "class"])[["gate_self", "gate_in", "gate_out", "gate_nbr"]].mean().round(3)
        L += ["\n## Mean gate weights on test nodes (averaged over seeds)\n", agg.reset_index().to_markdown(index=False)]
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
