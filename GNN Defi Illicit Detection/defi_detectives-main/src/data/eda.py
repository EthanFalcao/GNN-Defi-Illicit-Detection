import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA

from src.utils.load_data import load_tabular, load_graph

# ------------------------
# LOAD & BUILD DF
# ------------------------
print("Starting EDA...")
def build_df():
    X, y, tx_ids, timesteps, train_idx, test_idx = load_tabular()

    df = pd.DataFrame(X)
    df["label"] = y
    df["txId"] = tx_ids
    df["timesteps"] = timesteps

    # print dataset lengths
    print("Training and Test Lengths:")
    print(len(set(train_idx) & set(test_idx)))

    print("Train max timestep:", timesteps[train_idx].max())
    print("Test min timestep:", timesteps[test_idx].min())

    print("Train timestep range:", timesteps[train_idx].min(), timesteps[train_idx].max())
    print("Test timestep range:", timesteps[test_idx].min(), timesteps[test_idx].max())
    
    print("Overlap in time:",
          set(timesteps[train_idx]).intersection(set(timesteps[test_idx])))
    return df

# ------------------------
# BASIC OVERVIEW
# ------------------------
def basic_features(df):
    print("Shape:", df.shape)
    print("\nMissing values:", df.isnull().sum().sum())

    print("\nClass distribution:")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True))
    
    X_cols = df.drop(columns=["label", "txId", "timesteps"])

    print("\nFeature summary:")
    print(X_cols.describe().T.head())

    # correlation with label (top 10)
    corr = X_cols.select_dtypes(include=[np.number]).corrwith(df["label"]).sort_values()

    plt.figure(figsize=(8,5))
    corr.tail(10).plot(kind="barh")
    plt.title("Top positive correlations with fraud")
    plt.show()

    print("Checking feature seperability...")

    X = df.drop(columns=["label","txId","timesteps"])
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.scatter(X_2d[:,0], X_2d[:,1], c=df["label"], cmap="coolwarm")
    plt.title("PCA separation of fraud vs non-fraud")
    plt.show()


# ------------------------
# TIME ANALYSIS
# ------------------------
def time_analysis(df):
    print("\n====================")
    print(" TIME ANALYSIS")
    print("====================")

    time_counts = df["timesteps"].value_counts().sort_index()

    plt.figure(figsize=(10,4))
    sns.lineplot(x=time_counts.index, y=time_counts.values)
    plt.title("Transactions over time")
    plt.xlabel("Time step")
    plt.ylabel("Count")
    plt.show()

    fraud_rate = df.groupby("timesteps")["label"].mean()

    df.groupby("timesteps")["label"].value_counts(normalize=True)

    plt.figure(figsize=(10,4))
    sns.lineplot(x=fraud_rate.index, y=fraud_rate.values)
    plt.title("Fraud rate over time")
    plt.xlabel("Time step")
    plt.ylabel("Fraud rate")
    plt.show()

# ------------------------
# GRAPH ANALYSIS
# ------------------------
def graph_analysis(edge_index):
    print("\n====================")
    print(" GRAPH ANALYSIS")
    print("====================")

    G = nx.Graph()
    G.add_edges_from(edge_index.T)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    print("Avg degree:", sum(dict(G.degree()).values()) / G.number_of_nodes())

    # degree distribution
    degrees = [d for _, d in G.degree()]

    plt.figure(figsize=(6,4))
    sns.histplot(degrees, bins=50)
    plt.title("Node degree distribution")
    plt.show()


# ------------------------
# MAIN
# ------------------------
def main():
    df = build_df()

    edges = np.load("data/processed/graph/edge_index.npy")

    basic_features(df)
    time_analysis(df)
    graph_analysis(edges)


if __name__ == "__main__":
    main()
