"""Elliptic loader -> directed PyG graph with a temporal split.

Edge direction: elliptic_txs_edgelist.csv lists (txId1, txId2) meaning BTC flowed
from the output of txId1 into the input of txId2. We keep src=txId1 -> dst=txId2.
  in-neighbours of v  = predecessors (where v's funds came from)
  out-neighbours of v = successors   (where v's funds went)

Labels: 1 = illicit, 0 = licit, -1 = unknown.
Split (temporal, Weber et al. 2019 convention): train steps 1-34, test 35-49.
For early stopping we hold out steps 30-34 of the training period as validation,
so train = 1-29, val = 30-34, test = 35-49.
"""
from __future__ import annotations
import io, os, zipfile, glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

FILES = ["elliptic_txs_features.csv", "elliptic_txs_edgelist.csv", "elliptic_txs_classes.csv"]


def _open_sources(path: str):
    """Return dict name->file-like for the three CSVs, from a zip or a directory."""
    out = {}
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        for f in FILES:
            match = [n for n in zf.namelist() if n.endswith(f)]
            if not match:
                raise FileNotFoundError(f"{f} not in {path}")
            out[f] = zf.open(match[0])
    else:
        zips = glob.glob(os.path.join(path, "**", "*.zip"), recursive=True)
        for f in FILES:
            match = glob.glob(os.path.join(path, "**", f), recursive=True)
            if match:
                out[f] = open(match[0], "rb"); continue
            for z in zips:  # e.g. Kaggle per-file downloads: elliptic_txs_classes.csv.zip
                zf = zipfile.ZipFile(z)
                inner = [n for n in zf.namelist() if n.endswith(f)]
                if inner:
                    out[f] = zf.open(inner[0]); break
            if f not in out:
                raise FileNotFoundError(f"{f} not under {path} (plain or zipped)")
    return out


def load_elliptic(raw: str, cache: str | None = None, train_end=29, val_end=34) -> Data:
    if cache and os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    src = _open_sources(raw)
    feats = pd.read_csv(src[FILES[0]], header=None)
    edges = pd.read_csv(src[FILES[1]])
    classes = pd.read_csv(src[FILES[2]])

    feats = feats.rename(columns={0: "txId", 1: "time_step"})
    idx = {t: i for i, t in enumerate(feats["txId"].values)}
    classes = classes.set_index("txId").loc[feats["txId"].values]
    y = classes["class"].astype(str).map({"1": 1, "2": 0, "unknown": -1}).values

    s = edges.iloc[:, 0].map(idx).values
    d = edges.iloc[:, 1].map(idx).values
    edge_index = torch.tensor(np.vstack([s, d]), dtype=torch.long)

    x = torch.tensor(feats.loc[:, 2:].values, dtype=torch.float)
    # local features = first 93 after time_step (94 "local" incl. time step per paper)
    t = torch.tensor(feats["time_step"].values, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long), time=t)
    data.n_local = 93
    lab = data.y >= 0
    data.train_mask = lab & (t <= train_end)
    data.val_mask = lab & (t > train_end) & (t <= val_end)
    data.test_mask = lab & (t > val_end)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(data, cache)
    return data


def standardize(data: Data, local_only: bool = False) -> torch.Tensor:
    """z-score features using training-period statistics only (no test leakage)."""
    x = data.x[:, : data.n_local] if local_only else data.x
    ref = x[data.time <= 34]
    mu, sd = ref.mean(0), ref.std(0).clamp_min(1e-6)
    return (x - mu) / sd


def make_synthetic(path: str, n=6000, steps=49, seed=0):
    """Fake Elliptic with identical schema, for pipeline testing only."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(10**9, n, replace=False)
    ts = np.sort(rng.integers(1, steps + 1, n))
    cls = rng.choice(["1", "2", "unknown"], n, p=[0.02, 0.21, 0.77])
    X = rng.normal(size=(n, 165)) + (cls == "1")[:, None] * 0.3
    feats = pd.DataFrame(np.column_stack([ids, ts, X]))
    feats[0] = feats[0].astype(np.int64); feats[1] = feats[1].astype(int)
    e = []
    for tt in range(1, steps + 1):
        nodes = np.where(ts == tt)[0]
        for _ in range(int(len(nodes) * 1.15)):
            a, b = rng.choice(nodes, 2, replace=False)
            e.append((ids[a], ids[b]))
    os.makedirs(path, exist_ok=True)
    feats.to_csv(os.path.join(path, FILES[0]), header=False, index=False)
    pd.DataFrame(e, columns=["txId1", "txId2"]).to_csv(os.path.join(path, FILES[1]), index=False)
    pd.DataFrame({"txId": ids, "class": cls}).to_csv(os.path.join(path, FILES[2]), index=False)
