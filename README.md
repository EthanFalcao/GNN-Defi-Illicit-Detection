# Defi Detectives

## Expected raw files
Running `download_elliptic.py` will place these Kaggle Elliptic files in `data/raw/elliptic/`:
- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

## What the preprocessing pipeline creates
- **Baseline/tabular artifacts** in `data/processed/baseline/`
  - `elliptic_tabular.csv`
  - `X.npy`
  - `y.npy`
  - `tx_ids.npy`
  - `timesteps.npy`
- **Graph artifacts** in `data/processed/graph/`
  - `node_features.npy`
  - `labels.npy`
  - `edge_index.npy`
  - `node_id_map.csv`
  - `graph_data.pt` (PyTorch Geometric `Data` object if PyG is installed)
- **Fixed splits** in `data/processed/splits/`
  - `split_indices.npz`
  - `split_summary.json`

## Quickstart
```bash
python -m venv .venv 
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/data/preprocess_elliptic.py
```

## Download and preprocessing data
- Go to Kaggle, click on Profile -> Settings -> API Tokens -> Generate New API Token. This will generate a new API Token.
- Run
```bash 
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
```
- Paste this format:
```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
```
- Save the file using `CTRL + X`, then `Y`, and `Enter`.
- Fix permissions: `chmod 600 ~/.kaggle/kaggle.json`
- To download the raw Elliptic dataset, run `python src/data/download_elliptic.py`. This will place the expected CSV files in `data/raw/elliptic/`.
- To preprocess the raw data into baseline/tabular and graph artifacts, run `python src/data/preprocess_elliptic.py`. This will create the expected files in `data/processed/baseline/` and `data/processed/graph/`.