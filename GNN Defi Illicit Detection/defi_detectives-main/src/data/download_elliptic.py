import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "ellipticco/elliptic-data-set"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "elliptic"

def check_kaggle_installed() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not installed.\n"
            "Install it with: pip install kaggle"
        )


def check_kaggle_auth() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token = Path.home() / ".kaggle" / "access_token"
    env_token = os.getenv("KAGGLE_API_TOKEN")

    if kaggle_json.exists():
        return
    if access_token.exists():
        return
    if env_token:
        return

    raise RuntimeError(
        "Kaggle authentication not found.\n\n"
        "Use one of these:\n"
        "1) Place kaggle.json in ~/.kaggle/kaggle.json\n"
        "2) Place access token in ~/.kaggle/access_token\n"
        "3) Export KAGGLE_API_TOKEN in your environment\n"
    )


def download_dataset(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    expected_files = [
        RAW_DIR / "elliptic_txs_classes.csv",
        RAW_DIR / "elliptic_txs_edgelist.csv",
        RAW_DIR / "elliptic_txs_features.csv",
    ]

    if not force and all(p.exists() for p in expected_files):
        print("Elliptic dataset already exists. Skipping download.")
        return

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "--path",
        str(RAW_DIR),
        "--unzip",
    ]

    print(f"Downloading {DATASET_SLUG} into {RAW_DIR} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Kaggle download failed.")

    print("Download complete.")
    print("Files in directory:")
    for path in sorted(RAW_DIR.iterdir()):
        print(f" - {path.name}")


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    check_kaggle_installed()
    check_kaggle_auth()
    download_dataset(force=force_flag)