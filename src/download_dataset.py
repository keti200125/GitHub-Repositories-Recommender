"""Download the Kaggle GitHub repositories dataset with KaggleHub."""

import os
from pathlib import Path

import kagglehub
from dotenv import load_dotenv


DATASET_ID = "donbarbos/github-repos"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def configure_kaggle_credentials() -> None:
    """Load Kaggle credentials from the local .env file."""
    load_dotenv(ENV_PATH)

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username and key:
        print("Using Kaggle credentials from .env.")
        return

    print("Kaggle credentials were not found in .env.")
    print("Create a .env file from .env.example and set KAGGLE_USERNAME and KAGGLE_KEY.")
    print("Continuing without local credentials; KaggleHub may ask for authentication.")


def main() -> None:
    """Download the dataset and list the files in KaggleHub's local cache."""
    configure_kaggle_credentials()

    try:
        dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    except Exception as exc:
        print(f"Failed to download Kaggle dataset '{DATASET_ID}'.")
        print("Check your internet connection and Kaggle authentication credentials.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Dataset downloaded to: {dataset_path}")
    print("Files found:")

    files = sorted(path for path in dataset_path.rglob("*") if path.is_file())
    if not files:
        print("No files found in the downloaded dataset folder.")
        return

    for path in files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
