"""Thesis figures (static PNG + PDF) from results/*.json|csv."""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#257de0", "#f47230"      # categorical slots 1, 2
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_homophily(out):
    R = json.load(open(os.path.join(out, "homophily.json")))
    groups = [("All steps", R["per_class"]), ("Steps 1–34", R["per_class_steps_1_34"]), ("Steps 35–49", R["per_class_steps_35_49"])]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    w = 0.36
    for i, (lab, pc) in enumerate(groups):
        hi, ho = pc["illicit"]["h_in (mean over covered nodes)"], pc["illicit"]["h_out (mean over covered nodes)"]
        for dx, v, c in [(-w / 2 - 0.01, hi, BLUE), (w / 2 + 0.01, ho, ORANGE)]:
            ax.bar(i + dx, v, w, color=c, zorder=2)
            ax.text(i + dx, v + 0.015, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set_ylim(0, 0.85); ax.set_ylabel("Share of labelled neighbours\nthat are illicit")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.bar(0, 0, color=BLUE, label="Incoming (funding sources)"); ax.bar(0, 0, color=ORANGE, label="Outgoing (where funds go)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Illicit transactions: neighbour homophily by edge direction (Elliptic)", fontsize=10, color=INK, loc="left")
    save(fig, out, "fig_direction_homophily")


def fig_models(out):
    r = pd.read_csv(os.path.join(out, "runs.csv"))
    order = list(dict.fromkeys(r["model"]))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.38 * len(order) + 1.0))
    for feat, c, off, lab in [("LF", BLUE, 0.13, "Local features (93)"), ("AF", ORANGE, -0.13, "All features (165)")]:
        g = r[r.features == feat].groupby("model")["test_pr_auc"].agg(["mean", "std"]).reindex(order)
        y = [i + off for i in range(len(order))]
        ax.errorbar(g["mean"], y, xerr=g["std"], fmt="o", ms=6, color=c, ecolor=c, elinewidth=1.5, capsize=0, label=lab, zorder=3)
    ax.set_yticks(range(len(order)), order)
    ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_xlabel("Test PR-AUC, illicit class (mean ± sd over seeds)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    ax.set_title("Model comparison, temporal test steps 35–49", fontsize=10, color=INK, loc="left", pad=24)
    save(fig, out, "fig_model_prauc")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
    fig_homophily(a.out)
    if os.path.exists(os.path.join(a.out, "runs.csv")):
        fig_models(a.out)
