# Data Directory Guide

This file documents how to restore the dataset from `data.zip` and what is expected inside `data/`.

## Unzip `data.zip`

`data.zip` is shared via Teams and should be available to you. It contains the processed dataset files. Depending on where you placed `data.zip`, use one of the following commands to unzip it:

```bash
# Option A: if archive is at repo root
unzip -o data.zip -d data

# Option B: if archive is already inside data/
unzip -o data/data.zip -d data
```

After unzipping, `data/` should contain the structure shown below.

## Current `data/` contents

```text
data/
├── data.md
├── processed/
│   ├── baseline/
│   │   ├── X.npy
│   │   ├── elliptic_tabular.csv
│   │   ├── timesteps.npy
│   │   ├── tx_ids.npy
│   │   └── y.npy
│   ├── graph/
│   │   ├── edge_index.npy
│   │   ├── graph_data.pt
│   │   ├── labels.npy
│   │   ├── node_features.npy
│   │   ├── node_id_map.csv
│   │   ├── scaler_mean.npy
│   │   └── scaler_scale.npy
│   └── splits/
│       ├── split_indices.npz
│       └── split_summary.json
└── raw/
	├── elliptic/
	│   └── elliptic_bitcoin_dataset/
	│       ├── elliptic_txs_classes.csv
	│       ├── elliptic_txs_edgelist.csv
	│       └── elliptic_txs_features.csv
	└── ethereum_bigquery/
```

## Notes

- `raw/ethereum_bigquery/` exists but is currently empty.
- If your extracted files differ, re-check the unzip destination (`-d data`) and the archive layout.

