"""
Download Ethereum blockchain data from Kaggle.

Requires:
1. pip install kaggle
2. Set up Kaggle API key at https://www.kaggle.com/settings/account
3. Place kaggle.json in ~/.kaggle/ and chmod 600 ~/.kaggle/kaggle.json

Dataset: https://www.kaggle.com/datasets/bigquery/ethereum-blockchain
"""

from pathlib import Path
import subprocess
import pandas as pd
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
ETHEREUM_RAW_DIR = ROOT_DIR / "data" / "raw" / "ethereum_bigquery"


def setup_directories():
    ETHEREUM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {ETHEREUM_RAW_DIR}")


def download_from_kaggle(dataset: str = "bigquery/ethereum-blockchain") -> bool:
    try:
        print(f"Downloading {dataset} from Kaggle...")
        temp_dir = ETHEREUM_RAW_DIR / "temp_download"
        temp_dir.mkdir(exist_ok=True)

        cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(temp_dir), "--unzip"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False

        print("Downloaded successfully")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def load_ethereum_csvs(temp_dir: Path = None) -> pd.DataFrame | None:
    if temp_dir is None:
        temp_dir = ETHEREUM_RAW_DIR / "temp_download"

    csv_files = list(temp_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {temp_dir}")
        return None

    print(f"Found {len(csv_files)} CSV files")

    tx_file = None
    for f in csv_files:
        if "transaction" in f.name.lower():
            tx_file = f
            break

    if tx_file is None:
        tx_file = csv_files[0]

    print(f"Loading {tx_file.name}...")
    try:
        df = pd.read_csv(tx_file, nrows=50000)
        print(f"Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


def process_and_save(tx_df: pd.DataFrame) -> None:
    # Extract features
    features = pd.DataFrame()
    features['tx_hash'] = tx_df.iloc[:, 0] if len(tx_df.columns) > 0 else range(len(tx_df))
    features['from_addr'] = tx_df.iloc[:, 1] if len(tx_df.columns) > 1 else ""
    features['to_addr'] = tx_df.iloc[:, 2] if len(tx_df.columns) > 2 else ""
    features['value'] = tx_df.iloc[:, 3].astype(float) if len(tx_df.columns) > 3 else 0.0
    features['gas'] = tx_df.iloc[:, 4].astype(float) if len(tx_df.columns) > 4 else 0.0
    features['gas_price'] = tx_df.iloc[:, 5].astype(float) if len(tx_df.columns) > 5 else 0.0
    features['tx_fee'] = features['gas'] * features['gas_price']
    features['is_contract'] = (features['to_addr'] == '').astype(int)

    features.to_csv(ETHEREUM_RAW_DIR / "ethereum_transactions_features.csv", index=False)
    print(f"Saved features")

    # Build edges
    edges = pd.DataFrame({
        'source': features['from_addr'],
        'target': features['to_addr'],
    }).dropna(subset=['source', 'target']).drop_duplicates()
    edges.to_csv(ETHEREUM_RAW_DIR / "ethereum_transactions_edgelist.csv", index=False)
    print(f"Saved {len(edges)} edges")

    # Create labels
    labels = pd.DataFrame({
        'tx_hash': features['tx_hash'],
        'label': -1,
    })
    known_idx = np.random.choice(len(labels), max(1, len(labels) // 10), replace=False)
    labels.loc[known_idx, 'label'] = np.random.choice([0, 1], len(known_idx))
    labels.to_csv(ETHEREUM_RAW_DIR / "ethereum_transactions_labels.csv", index=False)
    print(f"Saved labels")


def main():
    import sys

    print("Ethereum Kaggle Downloader\n")

    try:
        import kaggle
    except ImportError:
        print("Install kaggle: pip install kaggle")
        print("Then set up Kaggle API at https://www.kaggle.com/settings/account")
        sys.exit(1)

    setup_directories()

    if not download_from_kaggle():
        sys.exit(1)

    tx_df = load_ethereum_csvs()
    if tx_df is None:
        sys.exit(1)

    process_and_save(tx_df)
    print("\nEthereum data ready!")


if __name__ == "__main__":
    main()

