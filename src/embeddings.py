"""Generate semantic embeddings for cleaned GitHub repositories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPOSITORIES_PATH = PROCESSED_DATA_DIR / "repositories_clean.csv"
EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "repository_embeddings.npy"
METADATA_PATH = PROCESSED_DATA_DIR / "repositories_embeddings.csv"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_repositories(path: Path = METADATA_PATH) -> pd.DataFrame:
    """Load repository metadata saved alongside the embeddings."""
    if not path.exists():
        raise FileNotFoundError(f"Repository metadata file not found: {path}")

    return pd.read_csv(path)


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> np.ndarray:
    """Load repository embeddings from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    return np.load(path)


def load_clean_repositories(path: Path = REPOSITORIES_PATH) -> pd.DataFrame:
    """Load the cleaned repository dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned repository dataset not found: {path}. "
            "Run python src/data_preprocessing.py first."
        )

    repositories = pd.read_csv(path)
    if "Description" not in repositories.columns:
        raise ValueError(f"Missing required Description column in: {path}")

    repositories = repositories.dropna(subset=["Description"]).copy()
    repositories["Description"] = repositories["Description"].astype(str)
    return repositories


def generate_embeddings(repositories: pd.DataFrame) -> np.ndarray:
    """Generate sentence-transformer embeddings from repository descriptions."""
    model = SentenceTransformer(MODEL_NAME)
    descriptions = repositories["Description"].tolist()
    return model.encode(
        descriptions,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def save_embeddings(repositories: pd.DataFrame, embeddings: np.ndarray) -> None:
    """Save embeddings and repository metadata to processed data files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    repositories.to_csv(METADATA_PATH, index=False)


def main() -> None:
    """Generate and save repository semantic embeddings."""
    try:
        repositories = load_clean_repositories()
        embeddings = generate_embeddings(repositories)
        save_embeddings(repositories, embeddings)
    except Exception as exc:
        print("Failed to generate repository embeddings.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Number of repositories: {len(repositories)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embeddings output: {EMBEDDINGS_PATH}")
    print(f"Metadata output: {METADATA_PATH}")


if __name__ == "__main__":
    main()
