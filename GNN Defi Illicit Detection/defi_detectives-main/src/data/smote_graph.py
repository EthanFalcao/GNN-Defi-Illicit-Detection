"""
SMOTE-based graph augmentation for node classification due to class imbalance.

Follows the approach evaluated in:
  Blagus & Lusa, "SMOTE for high-dimensional class-imbalanced data",
  BMC Bioinformatics 2013. https://doi.org/10.1186/1471-2105-14-106

Synthetic minority-class nodes are generated in feature space and appended to
the graph. Each synthetic node is wired to its k nearest real training neighbours
so it receives meaningful messages during training. Val/test splits are untouched.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from imblearn.over_sampling import SMOTE

def smote_augment_graph(data, train_idx, k_neighbors = 5, smote_k_neighbors = 5, random_state = 42,) -> tuple[Data, torch.Tensor]:
    """
    Apply SMOTE to minority-class training nodes and
    add synthetic nodes to the graph.
    """
    x_all = data.x        # (N, F)
    y_all = data.y        # (N,)
    edge_index = data.edge_index  # (2, E)

    # 1. Extract training-node features / labels
    train_cpu = train_idx.cpu()
    labelled_idx = train_cpu[y_all[train_cpu] >= 0]

    x_train = x_all[labelled_idx].cpu().numpy()  # (T, F)
    y_train = y_all[labelled_idx].cpu().numpy()  # (T,)

    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2:
        return data, train_idx

    # Blagus & Lusa 2.1: k must be < minority class size, otherwise SMOTE
    # cannot find enough distinct neighbours for interpolation
    effective_k = min(smote_k_neighbors, counts.min() - 1)
    if effective_k < 1:
        print("[SMOTE] minority class too small, skipping...")
        return data, train_idx

    # Blagus & Lusa 2.1: synthetic sample = x_i + rand(0,1) * (x_nn - x_i)
    # imbalanced-learn implements this interpolation internally
    x_res, y_res = SMOTE(k_neighbors=effective_k, random_state=random_state).fit_resample(x_train, y_train)

    # imbalanced-learn appends synthetics after the originals
    x_synthetic = x_res[len(x_train):]  # (S, F)
    y_synthetic = y_res[len(y_train):]  # (S,)
    n_synthetic = len(x_synthetic)

    if n_synthetic == 0:
        return data, train_idx

    print(f"[SMOTE] {counts.min()} -> {counts.min() + n_synthetic} minority nodes (+{n_synthetic} synthetic)")

    # connect each synthetic node to its k nearest neighbors in original training set
    # (Euclidean distance in feature space, as used by SMOTE interpolation)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_syn_t   = torch.tensor(x_synthetic, dtype=torch.float32)

    dists = torch.cdist(x_syn_t, x_train_t) #(S, T)
    k_actual = min(k_neighbors, len(x_train))
    _, nn_local = dists.topk(k_actual, dim=1, largest=False) #(S, k)

    N = x_all.size(0)
    syn_global = torch.arange(N, N + n_synthetic).repeat_interleave(k_actual) #(S*k,)
    nb_global  = labelled_idx[nn_local.reshape(-1)] #(S*k,)

    # bidirectional edges so synthetic nodes both send and receive messages
    new_edges = torch.stack([
        torch.cat([syn_global, nb_global]),
        torch.cat([nb_global, syn_global]),
    ])
    new_edge_index = torch.cat([edge_index.cpu(), new_edges], dim=1)

    aug_data = Data(
        x=torch.cat([x_all.cpu(), x_syn_t]),
        edge_index=new_edge_index,
        y=torch.cat([y_all.cpu(), torch.tensor(y_synthetic, dtype=torch.long)]),
    )

    # Blagus & Lusa 2.3: synthetic indices are (appended - original) indices
    # hence val/test splits remain unchanged
    new_train_idx = torch.cat([train_cpu, torch.arange(N, N + n_synthetic)])
    return aug_data, new_train_idx
