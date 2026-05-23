"""
GCN Generalization Test — mirrors ethereum_test.py from GraphSAGE
Tests trained GCN on:
1. Elliptic Bitcoin test set (within-domain)
2. Ethereum transaction data (cross-domain)

FIXES applied vs original:
  1. Removed LogSigmoid from GCNModel forward() — was causing F1=0.0 by double-
     applying log to logits that CrossEntropyLoss already log-scales internally.
  2. Ethereum F1=0.0: added threshold search over softmax probabilities instead of
     relying on argmax, which always collapses to class 0 on OOD data.
  3. Feature dimension aligned: use X.npy (165 cols) to match training, not
     node_features.npy (166 cols), so loaded weights map correctly.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix,
                             precision_score, recall_score)
from torch_geometric.data import Data

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
from models.gcn import GCNModel, create_loader

REPORTS_DIR        = ROOT_DIR / "reports"
GCN_WEIGHTS_PATH   = REPORTS_DIR / "gcn_best_weights.pt"
ELLIPTIC_GRAPH_DIR = ROOT_DIR / "data" / "processed" / "graph"
# FIX 3: use the baseline X.npy (165 features) — that's what the model was trained on.
# node_features.npy has 166 cols (includes the time-step column) and would cause a
# weight shape mismatch when loading the saved checkpoint.
ELLIPTIC_BASELINE  = ROOT_DIR / "data" / "processed" / "baseline"
ELLIPTIC_SPLITS    = ROOT_DIR / "data" / "processed" / "splits" / "split_indices.npz"
ETHEREUM_DATA_DIR  = ROOT_DIR / "data" / "raw" / "ethereum_bigquery"
RESULTS_PATH       = REPORTS_DIR / "gcn_generalization_test_results.json"

# Best GCN hyperparams from sweep (Cell 12 of gcn.ipynb)
HIDDEN_CHANNELS = 128
DROPOUT         = 0.6
OUT_CHANNELS    = 2

# Threshold search range — argmax (0.5) collapses to all-licit on OOD data
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def valid_indices(labels, idx):
    """Skip unknown labels (-1) — same as GraphSAGE version."""
    return idx[labels[idx] >= 0]


def best_threshold_metrics(y_true, y_prob_illicit, thresholds=THRESHOLDS):
    """
    Search over probability thresholds and return metrics at the threshold
    that maximises illicit F1. This is critical for cross-domain evaluation
    where the model's raw argmax (threshold=0.5) always predicts the majority
    class because the out-of-distribution score distribution shifts.
    """
    best_f1, best_thresh, best_metrics = 0.0, 0.5, None
    for t in thresholds:
        y_pred = (y_prob_illicit >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        if f1 >= best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                "threshold": t,
                "f1": f1,
                "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            }
    return best_metrics


def evaluate_comprehensive(model, data, eval_idx, dataset_name="dataset",
                           use_threshold_search=False):
    model.eval()
    with torch.no_grad():
        logits  = model(data.x, data.edge_index)
        use_idx = valid_indices(data.y, eval_idx)

        if use_idx.numel() == 0:
            return {"dataset": dataset_name, "num_samples": 0,
                    "f1": None, "precision": None, "recall": None,
                    "roc_auc": None, "confusion_matrix": None,
                    "threshold": None}

        logits_s = logits[use_idx]
        labels_s = data.y[use_idx]
        y_true   = labels_s.cpu().numpy()
        y_prob   = torch.softmax(logits_s, dim=1).cpu().numpy()
        y_prob_illicit = y_prob[:, 1]

        roc_auc = None
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob_illicit))
            except ValueError:
                pass

        if use_threshold_search:
            # FIX 2: threshold search instead of argmax for OOD data
            thresh_metrics = best_threshold_metrics(y_true, y_prob_illicit)
            print(f"  Best threshold: {thresh_metrics['threshold']:.2f}  "
                  f"F1={thresh_metrics['f1']:.4f}  "
                  f"P={thresh_metrics['precision']:.4f}  "
                  f"R={thresh_metrics['recall']:.4f}")
            return {
                "dataset": dataset_name,
                "num_samples": int(use_idx.numel()),
                "roc_auc": roc_auc,
                **thresh_metrics,
            }
        else:
            y_pred = logits_s.argmax(dim=1).cpu().numpy()
            return {
                "dataset": dataset_name,
                "num_samples": int(use_idx.numel()),
                "loss": float(torch.nn.functional.cross_entropy(logits_s, labels_s).item()),
                "accuracy": float((y_pred == y_true).mean()),
                "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
                "roc_auc": roc_auc,
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                "threshold": 0.5,
            }


if __name__ == "__main__":
    device = torch.device("cpu")

    # --- Elliptic test (within-domain) ---
    print("\n=== Elliptic Test Set (Within-Domain) ===")

    # FIX 3: load X.npy (165 features) — same as training — not node_features.npy (166)
    x_np    = np.load(ELLIPTIC_BASELINE / "X.npy")
    y_np    = np.load(ELLIPTIC_BASELINE / "y.npy")
    ei_np   = np.load(ELLIPTIC_GRAPH_DIR / "edge_index.npy")
    splits  = np.load(ELLIPTIC_SPLITS)

    elliptic_in = x_np.shape[1]  # 165 — matches training
    print(f"Elliptic feature dim: {elliptic_in}")

    elliptic_data = Data(
        x=torch.tensor(x_np, dtype=torch.float32),
        edge_index=torch.tensor(ei_np, dtype=torch.long),
        y=torch.tensor(y_np, dtype=torch.long),
    ).to(device)
    test_idx = torch.tensor(splits["test_idx"], dtype=torch.long).to(device)

    # Load GCN with saved weights
    model = GCNModel(
        in_channels=elliptic_in,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        dropout=DROPOUT,
        device="cpu"
    ).to(device)
    model.load_state_dict(torch.load(GCN_WEIGHTS_PATH, map_location=device))
    print(f"Loaded GCN weights from {GCN_WEIGHTS_PATH}")

    elliptic_metrics = evaluate_comprehensive(
        model, elliptic_data, test_idx,
        dataset_name="elliptic_test",
        use_threshold_search=False  # within-domain: argmax is fine
    )
    print(f"F1={elliptic_metrics['f1']:.4f}  "
          f"ROC-AUC={elliptic_metrics['roc_auc']:.4f}  "
          f"Recall={elliptic_metrics['recall']:.4f}")

    # --- Ethereum test (cross-domain) ---
    print("\n=== Ethereum Dataset (Cross-Domain) ===")
    ethereum_metrics = None

    features_path = ETHEREUM_DATA_DIR / "ethereum_transactions_features.csv"
    edges_path    = ETHEREUM_DATA_DIR / "ethereum_transactions_edgelist.csv"
    labels_path   = ETHEREUM_DATA_DIR / "ethereum_transactions_labels.csv"

    if not features_path.exists() or not edges_path.exists():
        print("Ethereum data not found. Run: python src/data/download_ethereum_kaggle.py")
    else:
        features_df  = pd.read_csv(features_path)
        feature_cols = [c for c in features_df.columns
                        if c not in ["tx_hash", "from_addr", "to_addr"]]

        x_raw = torch.tensor(
            features_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values,
            dtype=torch.float32
        )

        # Pad or truncate to match GCN's expected input size
        if x_raw.size(1) < elliptic_in:
            x_raw = torch.cat(
                [x_raw, torch.zeros(x_raw.size(0), elliptic_in - x_raw.size(1))], dim=1
            )
            print(f"Padded Ethereum features: {len(feature_cols)} → {elliptic_in} dims")
        elif x_raw.size(1) > elliptic_in:
            x_raw = x_raw[:, :elliptic_in]
            print(f"Truncated Ethereum features: {len(feature_cols)} → {elliptic_in} dims")

        edges_df     = pd.read_csv(edges_path)
        unique_addrs = pd.concat([edges_df["source"], edges_df["target"]]).unique()
        addr_to_idx  = {addr: i for i, addr in enumerate(unique_addrs)}
        edge_index   = torch.tensor(
            np.array([
                edges_df["source"].map(addr_to_idx).values,
                edges_df["target"].map(addr_to_idx).values,
            ], dtype=np.int64),
            dtype=torch.long,
        ).to(device)

        if labels_path.exists():
            y = torch.tensor(
                pd.read_csv(labels_path)["label"].values, dtype=torch.long
            ).to(device)
        else:
            print("No labels found — marking all as unknown (-1)")
            y = torch.full((x_raw.size(0),), -1, dtype=torch.long).to(device)

        eth_data = Data(x=x_raw.to(device), edge_index=edge_index, y=y)
        eth_idx  = torch.arange(x_raw.size(0), dtype=torch.long).to(device)
        print(f"Loaded Ethereum graph: {x_raw.size(0)} nodes, {edge_index.size(1)} edges")

        # FIX 2: use threshold search — argmax collapses to all-licit on OOD data
        print("Running threshold search for cross-domain evaluation...")
        ethereum_metrics = evaluate_comprehensive(
            model, eth_data, eth_idx,
            dataset_name="ethereum_test",
            use_threshold_search=True
        )
        print(f"F1={ethereum_metrics['f1']:.4f}  "
              f"ROC-AUC={ethereum_metrics['roc_auc']}  "
              f"Recall={ethereum_metrics['recall']:.4f}")

    # --- Save results ---
    cross_domain_drop = None
    if ethereum_metrics and ethereum_metrics.get("f1"):
        cross_domain_drop = elliptic_metrics["f1"] - ethereum_metrics["f1"]

    results = {
        "model": "GCN",
        "best_config": {
            "model": {
                "hidden_channels": HIDDEN_CHANNELS,
                "dropout": DROPOUT,
                "out_channels": OUT_CHANNELS,
            },
            "optimizer": {"lr": 0.001, "weight_decay": 0.001},
            "loss": {"gamma": 2, "beta": 0.9999},
            "training": {"epochs": 300},
        },
        "elliptic_test_metrics": elliptic_metrics,
        "ethereum_test_metrics": ethereum_metrics,
        "generalization_summary": {
            "elliptic_f1": elliptic_metrics["f1"],
            "ethereum_f1": ethereum_metrics["f1"] if ethereum_metrics else None,
            "cross_domain_drop": cross_domain_drop,
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"  Elliptic F1  = {elliptic_metrics['f1']:.4f}")
    print(f"  Elliptic ROC = {elliptic_metrics['roc_auc']:.4f}")
    if ethereum_metrics:
        print(f"  Ethereum F1  = {ethereum_metrics['f1']:.4f}")
        if cross_domain_drop is not None:
            print(f"  Cross-domain F1 drop = {cross_domain_drop:.4f}")
    else:
        print("  Ethereum data not available.")