"""Stage 3/4: train baselines + proposed model on Elliptic (temporal split).

Resumable: each finished (model, features, seed) run is appended to results/runs.csv
and skipped on re-run.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(__file__))
from data import load_elliptic, standardize
from models import Net, graph_inputs

MODELS = {
    # name: (kind, kwargs)
    "MLP": ("mlp", {}),
    "GCN": ("gcn", {}),
    "GAT": ("gat", {}),
    "SAGE": ("sage", {}),
    "DirGNN(a=0.5)": ("dirgnn", {"alpha": 0.5}),
    "Undir-Global": ("namix", {"directed": False, "node_gate": False}),
    "CAGNN-style (Undir-Node)": ("namix", {"directed": False, "node_gate": True}),
    "Dir-Global": ("namix", {"directed": True, "node_gate": False}),
    "NADir (Dir-Node, proposed)": ("namix", {"directed": True, "node_gate": True}),
}


def metrics(prob, y, thr):
    pred = (prob >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"pr_auc": average_precision_score(y, prob), "roc_auc": roc_auc_score(y, prob),
            "precision": p, "recall": r, "f1": f}


def best_threshold(prob, y):
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [precision_recall_fscore_support(y, (prob >= t).astype(int), average="binary", zero_division=0)[2] for t in grid]
    return float(grid[int(np.argmax(f1s))])


def evaluate(prob, data):
    y, t = data.y.numpy(), data.time.numpy()
    va, te = data.val_mask.numpy(), data.test_mask.numpy()
    thr = best_threshold(prob[va], y[va])
    out = {"val_pr_auc": average_precision_score(y[va], prob[va]), "thr": thr}
    for name, th in [("@0.5", 0.5), ("@val", thr)]:
        for k, v in metrics(prob[te], y[te], th).items():
            out[f"test_{k}{name}" if k not in ("pr_auc", "roc_auc") else f"test_{k}"] = v
    # dark-market shutdown after step 43 (Weber et al. 2019)
    for nm, m in [("pre43", te & (t <= 42)), ("post43", te & (t >= 43))]:
        out[f"test_f1@val_{nm}"] = metrics(prob[m], y[m], thr)["f1"]
    return out


def _sub(mask, X, ei, ei_und, logdeg):
    """Exact induced subgraph: Elliptic has no cross-time-step edges, so a time-window
    subgraph gives identical messages to the full graph for its nodes."""
    from torch_geometric.utils import subgraph
    idx = mask.nonzero().view(-1)
    e, _ = subgraph(idx, ei, relabel_nodes=True, num_nodes=X.size(0))
    eu, _ = subgraph(idx, ei_und, relabel_nodes=True, num_nodes=X.size(0))
    return idx, (X[idx], e, eu, logdeg[idx])


def train_gnn(name, data, X, ei, ei_und, logdeg, seed, epochs, patience, lr, wd, hid):
    torch.manual_seed(seed); np.random.seed(seed)
    kind, kw = MODELS[name]
    model = Net(kind, X.size(1), hid=hid, **kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    t = data.time
    tr_idx, tr_g = _sub(t <= 29, X, ei, ei_und, logdeg)
    va_idx, va_g = _sub((t >= 30) & (t <= 34), X, ei, ei_und, logdeg)
    te_idx, te_g = _sub(t >= 35, X, ei, ei_und, logdeg)
    y = data.y
    ytr = y[tr_idx]; trm = ytr >= 0
    yva = y[va_idx].numpy(); vam = yva >= 0
    w = torch.tensor([1.0, float((ytr[trm] == 0).sum() / (ytr[trm] == 1).sum())])  # class weights
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(*tr_g)[trm], ytr[trm], weight=w)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(*va_g), 1)[:, 1].numpy()
        ap = average_precision_score(yva[vam], pv[vam])
        if ap > best:
            best, bad, best_state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    prob = np.zeros(data.num_nodes, dtype=np.float32)
    with torch.no_grad():
        prob[va_idx.numpy()] = torch.softmax(model(*va_g), 1)[:, 1].numpy()
        prob[te_idx.numpy()] = torch.softmax(model(*te_g), 1)[:, 1].numpy()
    gates = None
    if kind == "namix":  # gates from the test-subgraph pass, mapped back to global node ids
        gates = []
        for c in model.convs:
            g = np.full((data.num_nodes, c.last_gates.shape[1]), np.nan, dtype=np.float32)
            g[te_idx.numpy()] = c.last_gates.numpy(); gates.append(g)
    return prob, ep + 1, gates


def train_rf(data, X, seed):
    from sklearn.ensemble import RandomForestClassifier
    tr = data.train_mask.numpy()
    rf = RandomForestClassifier(n_estimators=50, max_features=50, n_jobs=-1, random_state=seed)
    rf.fit(X.numpy()[tr], data.y.numpy()[tr])
    return rf.predict_proba(X.numpy())[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--cache", default="data/processed/elliptic.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="all"); ap.add_argument("--features", default="AF,LF")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--hid", type=int, default=64); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    data = load_elliptic(a.raw, a.cache)
    ei = data.edge_index
    ei_und, logdeg = graph_inputs(ei, data.num_nodes)
    names = (["RF"] + list(MODELS)) if a.models == "all" else a.models.split("|")
    runs_path = os.path.join(a.out, "runs.csv")
    done = set()
    if os.path.exists(runs_path):
        with open(runs_path) as f:
            done = {(r["model"], r["features"], r["seed"]) for r in csv.DictReader(f)}
    for feat in a.features.split(","):
        X = standardize(data, local_only=(feat == "LF"))
        for name in names:
            for seed in [int(s) for s in a.seeds.split(",")]:
                if (name, feat, str(seed)) in done:
                    continue
                t0 = time.time()
                gates = None
                if name == "RF":
                    prob, ep = train_rf(data, X, seed), 0
                else:
                    prob, ep, gates = train_gnn(name, data, X, ei, ei_und, logdeg, seed,
                                                a.epochs, a.patience, a.lr, a.wd, a.hid)
                row = {"model": name, "features": feat, "seed": seed, "epochs": ep,
                       "secs": round(time.time() - t0, 1), **evaluate(prob, data)}
                new = not os.path.exists(runs_path)
                with open(runs_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
                if gates is not None:
                    save_gates(gates, data, name, feat, seed, a.out)
                print(f"{feat} {name:28s} s{seed} ep{ep:3d} {row['secs']:6.1f}s "
                      f"PR-AUC {row['test_pr_auc']:.3f} F1@val {row['test_f1@val']:.3f}", flush=True)


def save_gates(gates, data, name, feat, seed, out):
    """Mean gate weights on test nodes, split by class: are illicit nodes routed differently?"""
    y, te = data.y.numpy(), data.test_mask.numpy()
    rows = []
    for li, g in enumerate(gates):
        for c, cn in [(1, "illicit"), (0, "licit")]:
            m = te & (y == c)
            rows.append({"model": name, "features": feat, "seed": seed, "layer": li, "class": cn,
                         **{f"gate_{k}": float(v) for k, v in zip(["self", "in", "out"] if g.shape[1] == 3 else ["self", "nbr"], g[m].mean(0))}})
    path = os.path.join(out, "gates.csv"); new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "features", "seed", "layer", "class", "gate_self", "gate_in", "gate_out", "gate_nbr"])
        new and w.writeheader(); [w.writerow(r) for r in rows]


if __name__ == "__main__":
    main()
