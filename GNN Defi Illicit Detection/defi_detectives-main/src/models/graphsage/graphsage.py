"""
GraphSage implementation for elliptic dataset. Main model file.

Reference: https://arxiv.org/abs/1706.02216

This version uses GraphConv instead of following a complete paper implementation.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_DIR = ROOT_DIR / "data" / "processed" / "graph"
DEFAULT_SPLITS_PATH = ROOT_DIR / "data" / "processed" / "splits" / "split_indices.npz"


def load_data(graph_dir=DEFAULT_GRAPH_DIR, splits_path=DEFAULT_SPLITS_PATH):
    x = np.load(graph_dir / "node_features.npy")
    y = np.load(graph_dir / "labels.npy")
    edge_index = np.load(graph_dir / "edge_index.npy")
    splits = np.load(splits_path)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )
    train_idx = torch.tensor(splits["train_idx"],dtype=torch.long)
    val_idx = torch.tensor(splits["val_idx"],dtype=torch.long)
    test_idx  = torch.tensor(splits["test_idx"],dtype=torch.long)

    return data, train_idx, val_idx, test_idx


def valid_indices(labels, idx):
    """Elliptic marks unknown nodes as -1; skip them during loss/eval."""
    return idx[labels[idx] >= 0]


def compute_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Inverse-frequency class weights so the minority (illicit) class
    gets proportionally higher loss contribution.

    Suggested by Claude.

    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights


class GraphSage(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def train_one_epoch(self, data, train_idx, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        logits = self.forward(data.x, data.edge_index)
        use_idx = valid_indices(data.y, train_idx)
        loss = criterion(logits[use_idx], data.y[use_idx])
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self, data, eval_idx, criterion):
        self.eval()
        logits = self.forward(data.x, data.edge_index)
        use_idx = valid_indices(data.y, eval_idx)
        loss = criterion(logits[use_idx], data.y[use_idx]).item()
        pred = logits[use_idx].argmax(dim=1).cpu().numpy()
        true = data.y[use_idx].cpu().numpy()
        f1 = f1_score(true, pred, pos_label=1, average="binary", zero_division=0)
        return float(loss), float(f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'mps', 'cuda', or 'cpu'")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    # Auto-detect best available device (claude given code)
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using NVIDIA GPU (CUDA)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    data, train_idx, val_idx, test_idx = load_data()
    data = data.to(device)
    train_idx = train_idx.to(device)
    val_idx   = val_idx.to(device)
    test_idx  = test_idx.to(device)

    model = GraphSage(
        in_channels=data.x.size(-1),
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    labelled_train = valid_indices(data.y, train_idx)
    class_weights = compute_class_weights(data.y[labelled_train].cpu()).to(device)
    print(f"Class weights (licit, illicit): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0
    best_state  = None

    for epoch in range(1, args.epochs + 1):
        train_loss = model.train_one_epoch(data, train_idx, optimizer, criterion)
        val_loss, val_f1 = model.evaluate(data, val_idx, criterion)

        # Checkpoints on F1 instead of loss or accuracy, since F1 is more meaningful for imbalanced data
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_f1={val_f1:.4f}"
            )

    model.load_state_dict(best_state)
    test_loss, test_f1 = model.evaluate(data, test_idx, criterion)
    print(f"\nbest val_f1={best_val_f1:.4f}  |  test_loss={test_loss:.4f}  test_f1={test_f1:.4f}")