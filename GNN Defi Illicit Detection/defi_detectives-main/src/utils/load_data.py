import numpy as np

BASELINE_PATH = "data/processed/baseline"
GRAPH_PATH = "data/processed/graph"
SPLIT_PATH = "data/processed/splits"

# use this for Xgboost and RF
def load_tabular():
    X = np.load(f"{BASELINE_PATH}/X.npy")
    y = np.load(f"{BASELINE_PATH}/y.npy")
    tx_ids = np.load(f"{BASELINE_PATH}/tx_ids.npy", allow_pickle=True)
    timesteps = np.load(f"{BASELINE_PATH}/timesteps.npy")

    splits = np.load(f"{SPLIT_PATH}/split_indices.npz")
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

    return X, y, tx_ids, timesteps, train_idx, test_idx

# use this to load data when building graph models (link prediction)
def load_graph():
    edge_index = np.load(f"{GRAPH_PATH}/edge_index.npy")
    node_features = np.load(f"{GRAPH_PATH}/node_features.npy")
    labels = np.load(f"{GRAPH_PATH}/labels.npy")

    return edge_index, node_features, labels
