import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from torch_geometric.data import Data

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
from models.graphsage.graphsage import GraphSage, load_data, valid_indices

REPORTS_DIR       = ROOT_DIR / "reports"
BEST_CONFIG_PATH  = REPORTS_DIR / "graphsage_best_config.yaml"
BEST_WEIGHTS_PATH = REPORTS_DIR / "graphsage_best_weights.pt"
ELLIPTIC_GRAPH_DIR = ROOT_DIR / "data" / "processed" / "graph"
ELLIPTIC_SPLITS    = ROOT_DIR / "data" / "processed" / "splits" / "split_indices.npz"
ETHEREUM_DATA_DIR  = ROOT_DIR / "data" / "raw" / "ethereum_bigquery"
RESULTS_PATH       = REPORTS_DIR / "generalization_test_results.json"


def evaluate_comprehensive(model, data, eval_idx, dataset_name="dataset"):
    model.eval()
    with torch.no_grad():
        logits  = model(data.x, data.edge_index)
        use_idx = valid_indices(data.y, eval_idx)

        if use_idx.numel() == 0:
            return {"dataset": dataset_name, "num_samples": 0,
                    "loss": None, "accuracy": None, "f1": None,
                    "precision": None, "recall": None, "roc_auc": None, "confusion_matrix": None}

        logits_s = logits[use_idx]
        labels_s = data.y[use_idx]
        y_true   = labels_s.cpu().numpy()
        y_pred   = logits_s.argmax(dim=1).cpu().numpy()
        y_prob   = torch.softmax(logits_s, dim=1).cpu().numpy()

        roc_auc = None
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            except ValueError:
                pass

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
        }


if __name__ == "__main__":
    device = torch.device("cpu")

    # load best config + weights from sweep
    with open(BEST_CONFIG_PATH) as f:
        best_cfg = yaml.safe_load(f)

    mcfg = best_cfg["config"]["model"]
    print(f"Config: hidden={mcfg['hidden_channels']}, dropout={mcfg['dropout']}, lr={best_cfg['config']['optimizer']['lr']}")

    # --- Elliptic test (within-domain) ---
    print("\n=== Elliptic Test Set (Within-Domain) ===")
    data, _, _, test_idx = load_data(graph_dir=ELLIPTIC_GRAPH_DIR, splits_path=ELLIPTIC_SPLITS)
    data     = data.to(device)
    test_idx = test_idx.to(device)

    elliptic_in = data.x.size(-1)
    model = GraphSage(in_channels=elliptic_in,
                      hidden_channels=mcfg.get("hidden_channels", 64),
                      out_channels=mcfg.get("out_channels", 2),
                      dropout=mcfg.get("dropout", 0.5)).to(device)
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device))

    elliptic_metrics = evaluate_comprehensive(model, data, test_idx, dataset_name="elliptic_test")
    print(f"F1={elliptic_metrics['f1']:.4f}  ROC-AUC={elliptic_metrics['roc_auc']:.4f}  Recall={elliptic_metrics['recall']:.4f}")

    # --- Ethereum test (cross-domain) ---
    print("\n=== Ethereum Dataset (Cross-Domain) ===")
    ethereum_metrics = None

    features_path = ETHEREUM_DATA_DIR / "ethereum_transactions_features.csv"
    edges_path = ETHEREUM_DATA_DIR / "ethereum_transactions_edgelist.csv"
    labels_path = ETHEREUM_DATA_DIR / "ethereum_transactions_labels.csv"

    if not features_path.exists() or not edges_path.exists():
        print("Ethereum data not found. Run: python src/data/download_ethereum_kaggle.py")
    else:
        features_df = pd.read_csv(features_path)
        feature_cols = [c for c in features_df.columns if c not in ["tx_hash", "from_addr", "to_addr"]]
        x_raw = torch.tensor(features_df[feature_cols].values, dtype=torch.float32)

        # pad/truncate to match the Elliptic model's input size
        if x_raw.size(1) < elliptic_in:
            x_raw = torch.cat([x_raw, torch.zeros(x_raw.size(0), elliptic_in - x_raw.size(1))], dim=1)
            print(f"Padded Ethereum features to {elliptic_in} dims")
        elif x_raw.size(1) > elliptic_in:
            x_raw = x_raw[:, :elliptic_in]
            print(f"Truncated Ethereum features to {elliptic_in} dims")

        edges_df = pd.read_csv(edges_path)
        unique_addrs = pd.concat([edges_df["source"], edges_df["target"]]).unique()
        addr_to_idx = {addr: i for i, addr in enumerate(unique_addrs)}
        edge_index = torch.tensor(
            np.array([edges_df["source"].map(addr_to_idx).values,
                      edges_df["target"].map(addr_to_idx).values], dtype=np.int64),
            dtype=torch.long,
        ).to(device)

        if labels_path.exists():
            y = torch.tensor(pd.read_csv(labels_path)["label"].values, dtype=torch.long).to(device)
        else:
            print("No labels found, marking all as unknown (-1)")
            y = torch.full((x_raw.size(0),), -1, dtype=torch.long).to(device)

        eth_data = Data(x=x_raw.to(device), edge_index=edge_index, y=y)
        eth_idx = torch.arange(x_raw.size(0), dtype=torch.long).to(device)
        print(f"Loaded Ethereum graph: {x_raw.size(0)} nodes, {edge_index.size(1)} edges")

        ethereum_metrics = evaluate_comprehensive(model, eth_data, eth_idx, dataset_name="ethereum_test")
        print(f"F1={ethereum_metrics['f1']:.4f}  ROC-AUC={ethereum_metrics['roc_auc']}  Recall={ethereum_metrics['recall']:.4f}")

    # --- save results ---
    cross_domain_drop = None
    if ethereum_metrics and ethereum_metrics.get("f1"):
        cross_domain_drop = elliptic_metrics["f1"] - ethereum_metrics["f1"]

    results = {
        "best_trial": best_cfg.get("trial"),
        "best_config": best_cfg.get("config"),
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
    print(f"  Elliptic  F1={elliptic_metrics['f1']:.4f}  ROC-AUC={elliptic_metrics['roc_auc']:.4f}  Recall={elliptic_metrics['recall']:.4f}")
    if ethereum_metrics:
        print(f"  Ethereum  F1={ethereum_metrics['f1']:.4f}")
        print(f"  Cross-domain F1 drop: {cross_domain_drop:.4f}" if cross_domain_drop else "  Cross-domain F1 drop: N/A")
    else:
        print("  Ethereum data not available.")
