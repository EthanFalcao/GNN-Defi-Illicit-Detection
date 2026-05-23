"""
Preprocess the raw Elliptic Bitcoin dataset into numpy/PyTorch artifacts
ready for model training. Creates train/val/test splits for training.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SEED = 42
RAW_DIR = ROOT / "data/raw/elliptic/elliptic_bitcoin_dataset"
BASELINE_DIR = ROOT / "data/processed/baseline"
GRAPH_DIR = ROOT / "data/processed/graph"
SPLITS_DIR = ROOT / "data/processed/splits"


if __name__ == "__main__":
    # check raw files exist
    required_files = [
        RAW_DIR / "elliptic_txs_features.csv",
        RAW_DIR / "elliptic_txs_edgelist.csv",
        RAW_DIR / "elliptic_txs_classes.csv",
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        print("Missing files:\n" + "\n".join(f"  {p}" for p in missing))
        sys.exit(1)

    # load features
    features_df = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None)
    feature_cols = [f"feature_{i}" for i in range(features_df.shape[1] - 2)]
    features_df.columns = ["txId", "time_step"] + feature_cols

    # load labels (1=illicit, 2=licit, unknown=-1)
    classes_df = pd.read_csv(RAW_DIR / "elliptic_txs_classes.csv")
    classes_df["label"] = classes_df["class"].map({"1": 1, "2": 0, 1: 1, 2: 0, "unknown": -1}).fillna(-1).astype(int)

    # load edges
    edges_df = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")
    edges_df = edges_df.rename(columns={"txId1": "source", "txId2": "target"})

    # merge features + labels
    df = features_df.merge(classes_df[["txId", "label"]], on="txId", how="left")
    df["label"] = df["label"].fillna(-1).astype(int)

    # clean up inf/nan before scaling
    df[feature_cols] = (df[feature_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(df[feature_cols].median()))

    # standardize features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values)

    tx_ids        = df["txId"].to_numpy()
    timesteps     = df["time_step"].to_numpy(dtype=np.int64)
    labels        = df["label"].to_numpy(dtype=np.int64)
    node_features = df[feature_cols].to_numpy(dtype=np.float32)

    # build edge index (map txIds to node indices)
    id_to_idx  = {tx: i for i, tx in enumerate(tx_ids)}
    edges_df   = edges_df[edges_df["source"].isin(id_to_idx) & edges_df["target"].isin(id_to_idx)]
    edge_index = np.vstack([
        edges_df["source"].map(id_to_idx).to_numpy(dtype=np.int64),
        edges_df["target"].map(id_to_idx).to_numpy(dtype=np.int64),
    ])

    # stratified 70/15/15 split on labelled nodes only
    known_idx = np.where(labels != -1)[0]
    train_idx, temp_idx = train_test_split(known_idx, test_size=0.30, stratify=labels[known_idx], random_state=SEED)
    val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=SEED)
    train_idx = np.sort(train_idx)
    val_idx   = np.sort(val_idx)
    test_idx  = np.sort(test_idx)

    # save everything
    for d in [BASELINE_DIR, GRAPH_DIR, SPLITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # tabular artifacts for RF/XGB
    tabular_df = pd.DataFrame(node_features, columns=feature_cols)
    tabular_df.insert(0, "txId", tx_ids)
    tabular_df.insert(1, "time_step", timesteps)
    tabular_df["label"] = labels
    tabular_df.to_csv(BASELINE_DIR / "elliptic_tabular.csv", index=False)
    np.save(BASELINE_DIR / "X.npy", node_features)
    np.save(BASELINE_DIR / "y.npy", labels)
    np.save(BASELINE_DIR / "tx_ids.npy", tx_ids)
    np.save(BASELINE_DIR / "timesteps.npy", timesteps)

    # graph artifacts for GNN
    np.save(GRAPH_DIR / "node_features.npy", node_features)
    np.save(GRAPH_DIR / "labels.npy", labels)
    np.save(GRAPH_DIR / "edge_index.npy", edge_index)
    np.save(GRAPH_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(GRAPH_DIR / "scaler_scale.npy", scaler.scale_)
    pd.DataFrame({"txId": tx_ids, "node_idx": np.arange(len(tx_ids))}).to_csv(
        GRAPH_DIR / "node_id_map.csv", index=False
    )

    # save as PyG Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
        train_idx=torch.tensor(train_idx, dtype=torch.long),
        val_idx=torch.tensor(val_idx, dtype=torch.long),
        test_idx=torch.tensor(test_idx, dtype=torch.long),
    )
    torch.save(data, GRAPH_DIR / "graph_data.pt")

    # save splits
    np.savez(SPLITS_DIR / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # summary stats
    summary = {
        "seed": SEED,
        "num_nodes": int(node_features.shape[0]),
        "num_features": int(node_features.shape[1]),
        "num_edges": int(edge_index.shape[1]),
        "splits": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "label_counts": {
            "train": {"licit": int((labels[train_idx] == 0).sum()), "illicit": int((labels[train_idx] == 1).sum())},
            "val":   {"licit": int((labels[val_idx] == 0).sum()),   "illicit": int((labels[val_idx] == 1).sum())},
            "test":  {"licit": int((labels[test_idx] == 0).sum()),  "illicit": int((labels[test_idx] == 1).sum())},
        },
    }
    with open(SPLITS_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {node_features.shape[0]} nodes, {edge_index.shape[1]} edges")
    print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
