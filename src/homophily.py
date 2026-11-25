"""Stage 2: does heterophily in Elliptic depend on edge direction?

All label-based statistics use only edges whose two endpoints are labelled
(unknown = -1 is excluded); coverage is reported alongside every number.

Outputs results/homophily.json and results/homophily.md
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import load_elliptic

CLS = {0: "licit", 1: "illicit"}


def edge_homophily(ys, yd):
    return float((ys == yd).mean())


def adjusted_homophily(ys, yd, k=2):
    """Platonov et al. (2023) adjusted homophily on the undirected labelled subgraph."""
    h = edge_homophily(ys, yd)
    deg = np.bincount(np.concatenate([ys, yd]), minlength=k).astype(float)
    p = deg / deg.sum()
    return float((h - (p ** 2).sum()) / (1 - (p ** 2).sum()))


def compat(ys, yd, k=2):
    """C[a,b] = P(y_dst=b | y_src=a): where does money from class a go?"""
    M = np.zeros((k, k))
    np.add.at(M, (ys, yd), 1)
    return M, M / M.sum(1, keepdims=True)


def node_dir_homophily(src, dst, y, n):
    """Per node: fraction of labelled in-/out-neighbours sharing its label."""
    lab = (y[src] >= 0) & (y[dst] >= 0)
    s, d = src[lab], dst[lab]
    same = (y[s] == y[d]).astype(float)
    out_same = np.bincount(s, same, n); out_cnt = np.bincount(s, minlength=n)
    in_same = np.bincount(d, same, n); in_cnt = np.bincount(d, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return in_same / in_cnt, in_cnt, out_same / out_cnt, out_cnt


def class_summary(src, dst, y, n):
    h_in, c_in, h_out, c_out = node_dir_homophily(src, dst, y, n)
    deg_in = np.bincount(dst, minlength=n); deg_out = np.bincount(src, minlength=n)
    res = {}
    for c, name in CLS.items():
        m = y == c
        res[name] = {
            "n_nodes": int(m.sum()),
            "mean_in_degree": float(deg_in[m].mean()),
            "mean_out_degree": float(deg_out[m].mean()),
            "share_no_in_edges": float((deg_in[m] == 0).mean()),
            "share_no_out_edges": float((deg_out[m] == 0).mean()),
            "coverage_in (>=1 labelled in-nbr)": float((c_in[m] > 0).mean()),
            "coverage_out (>=1 labelled out-nbr)": float((c_out[m] > 0).mean()),
            "h_in (mean over covered nodes)": float(np.nanmean(h_in[m])),
            "h_out (mean over covered nodes)": float(np.nanmean(h_out[m])),
        }
    return res


def flip_test(src, dst, y, n, cls=1, perms=1000, seed=0):
    """Null: direction carries no class information. Randomly reverse each edge
    (p=0.5) and recompute h_out - h_in for nodes of class `cls`."""
    def stat(s, d):
        h_in, _, h_out, _ = node_dir_homophily(s, d, y, n)
        m = y == cls
        return np.nanmean(h_out[m]) - np.nanmean(h_in[m])
    obs = stat(src, dst)
    rng = np.random.default_rng(seed)
    null = np.empty(perms)
    for i in range(perms):
        f = rng.random(len(src)) < 0.5
        s = np.where(f, dst, src); d = np.where(f, src, dst)
        null[i] = stat(s, d)
    p = float((np.abs(null) >= abs(obs)).mean())
    return {"observed_h_out_minus_h_in": float(obs), "null_mean": float(null.mean()),
            "null_std": float(null.std()), "p_value_two_sided": p, "perms": perms}


def run(data, out_dir, perms=1000):
    y = data.y.numpy(); n = len(y)
    src, dst = data.edge_index.numpy()
    t = data.time.numpy()
    lab = (y[src] >= 0) & (y[dst] >= 0)
    ys, yd = y[src][lab], y[dst][lab]
    M, C = compat(ys, yd)
    R = {
        "n_nodes": n, "n_edges": int(len(src)),
        "n_labelled_edges": int(lab.sum()), "labelled_edge_share": float(lab.mean()),
        "label_counts": {"illicit": int((y == 1).sum()), "licit": int((y == 0).sum()), "unknown": int((y == -1).sum())},
        "cross_timestep_edges": int((t[src] != t[dst]).sum()),
        "edge_homophily": edge_homophily(ys, yd),
        "adjusted_homophily_undirected": adjusted_homophily(ys, yd),
        "edge_counts_src_to_dst [licit,illicit]": M.astype(int).tolist(),
        "P(dst class | src class)": C.tolist(),
        "P(src class | dst class)": (M.T / M.T.sum(1, keepdims=True)).tolist(),
        "per_class": class_summary(src, dst, y, n),
        "flip_test_illicit": flip_test(src, dst, y, n, 1, perms),
        "flip_test_licit": flip_test(src, dst, y, n, 0, perms),
    }
    # by period: does the structure shift between train and test periods?
    for name, (lo, hi) in {"steps_1_34": (1, 34), "steps_35_49": (35, 49)}.items():
        m = (t[src] >= lo) & (t[src] <= hi)
        R[f"per_class_{name}"] = class_summary(src[m], dst[m], np.where((t >= lo) & (t <= hi), y, -1), n)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "homophily.json"), "w") as f:
        json.dump(R, f, indent=2)
    with open(os.path.join(out_dir, "homophily.md"), "w") as f:
        f.write(to_md(R))
    return R


def to_md(R):
    L = ["# Direction-aware homophily analysis (Elliptic)\n",
         f"Nodes {R['n_nodes']:,}; edges {R['n_edges']:,}; edges with both ends labelled "
         f"{R['n_labelled_edges']:,} ({100*R['labelled_edge_share']:.1f}%). "
         f"Cross-time-step edges: {R['cross_timestep_edges']}.\n",
         f"Edge homophily {R['edge_homophily']:.3f}; adjusted homophily (undirected) "
         f"{R['adjusted_homophily_undirected']:.3f}.\n",
         "## Where does money go? P(dst class | src class)\n",
         "| src \\ dst | licit | illicit |", "|---|---|---|"]
    for i, nm in enumerate(["licit", "illicit"]):
        L.append(f"| {nm} | {R['P(dst class | src class)'][i][0]:.3f} | {R['P(dst class | src class)'][i][1]:.3f} |")
    L += ["\n## Where does money come from? P(src class | dst class)\n",
          "| dst \\ src | licit | illicit |", "|---|---|---|"]
    for i, nm in enumerate(["licit", "illicit"]):
        L.append(f"| {nm} | {R['P(src class | dst class)'][i][0]:.3f} | {R['P(src class | dst class)'][i][1]:.3f} |")
    L += ["\n## Per-class directional homophily (all time steps)\n"]
    keys = list(R["per_class"]["illicit"].keys())
    L += ["| metric | illicit | licit |", "|---|---|---|"]
    for k in keys:
        a, b = R["per_class"]["illicit"][k], R["per_class"]["licit"][k]
        fmt = (lambda v: f"{v:,}") if isinstance(a, int) else (lambda v: f"{v:.3f}")
        L.append(f"| {k} | {fmt(a)} | {fmt(b)} |")
    L += ["\n## Edge-flip permutation test (H0: direction carries no class information)\n",
          "| class | h_out - h_in (observed) | null mean ± sd | p (two-sided) |", "|---|---|---|---|"]
    for c in ["illicit", "licit"]:
        ft = R[f"flip_test_{c}"]
        L.append(f"| {c} | {ft['observed_h_out_minus_h_in']:+.3f} | {ft['null_mean']:+.3f} ± {ft['null_std']:.3f} | {ft['p_value_two_sided']:.4f} |")
    L += ["\n## Train period (1-34) vs test period (35-49)\n",
          "| metric | illicit 1-34 | illicit 35-49 | licit 1-34 | licit 35-49 |", "|---|---|---|---|---|"]
    for k in ["h_in (mean over covered nodes)", "h_out (mean over covered nodes)", "mean_in_degree", "mean_out_degree"]:
        a = R["per_class_steps_1_34"]; b = R["per_class_steps_35_49"]
        L.append(f"| {k} | {a['illicit'][k]:.3f} | {b['illicit'][k]:.3f} | {a['licit'][k]:.3f} | {b['licit'][k]:.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--cache", default="data/processed/elliptic.pt")
    ap.add_argument("--out", default="results"); ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    R = run(load_elliptic(a.raw, a.cache), a.out, a.perms)
    print(to_md(R))
