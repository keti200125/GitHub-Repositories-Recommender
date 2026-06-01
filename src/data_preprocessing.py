"""Clean the raw GitHub repositories dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub
import pandas as pd


DATASET_ID = "donbarbos/github-repos"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_PATH = PROCESSED_DATA_DIR / "repositories_clean.csv"
KAGGLEHUB_CACHE_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / "donbarbos" / "github-repos"

REQUIRED_COLUMNS = [
    "Name",
    "Description",
    "URL",
    "CreatedAt",
    "UpdatedAt",
    "Stars",
    "Forks",
    "Issues",
]

COLUMN_NAME_MAP = {
    "".join(character for character in column.lower() if character.isalnum()): column
    for column in REQUIRED_COLUMNS
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Clean the GitHub repositories dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a raw GitHub repositories CSV file.",
    )
    return parser.parse_args()


def canonicalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize known column name variants to the project column names."""
    existing_columns = set(dataframe.columns)
    rename_columns: dict[str, str] = {}

    for column in dataframe.columns:
        normalized = "".join(character for character in column.lower() if character.isalnum())
        canonical_column = COLUMN_NAME_MAP.get(normalized)

        if canonical_column and column != canonical_column and canonical_column not in existing_columns:
            rename_columns[column] = canonical_column
            existing_columns.add(canonical_column)

    return dataframe.rename(columns=rename_columns)


def find_csv_files(folder: Path) -> list[Path]:
    """Return CSV files inside a folder, sorted by size descending."""
    if not folder.exists():
        return []

    csv_files = [path for path in folder.rglob("*.csv") if path.is_file()]
    return sorted(csv_files, key=lambda path: path.stat().st_size, reverse=True)


def find_downloaded_dataset_csv() -> Path:
    """Find a likely CSV file from the KaggleHub cache or local raw data folder."""
    candidate_dirs = [KAGGLEHUB_CACHE_DIR, RAW_DATA_DIR]

    for folder in candidate_dirs:
        csv_files = find_csv_files(folder)
        if csv_files:
            return csv_files[0]

    try:
        kagglehub_folder = Path(kagglehub.dataset_download(DATASET_ID))
    except Exception as exc:
        print(f"Could not locate the KaggleHub dataset cache automatically: {exc}")
    else:
        csv_files = find_csv_files(kagglehub_folder)
        if csv_files:
            return csv_files[0]

    raise FileNotFoundError(
        "No CSV file was found automatically. "
        "Pass one explicitly with: python src/data_preprocessing.py --input path/to/file.csv"
    )


def validate_columns(dataframe: pd.DataFrame, input_path: Path) -> None:
    """Ensure the expected dataset columns are present."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {input_path}: {missing}")


def clean_repositories(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean repository rows and return the top repositories by stars."""
    cleaned = dataframe.loc[:, REQUIRED_COLUMNS].copy()

    cleaned["Name"] = cleaned["Name"].astype("string").str.strip()
    cleaned["Description"] = cleaned["Description"].astype("string").str.strip()
    cleaned = cleaned.dropna(subset=["Name", "Description"])
    cleaned = cleaned[(cleaned["Name"] != "") & (cleaned["Description"] != "")]

    cleaned = cleaned.drop_duplicates(subset=["Name"])

    for column in ["Stars", "Forks", "Issues"]:
        values = cleaned[column].astype("string").str.replace(",", "", regex=False)
        cleaned[column] = pd.to_numeric(values, errors="coerce").fillna(0).astype("int64")

    for column in ["CreatedAt", "UpdatedAt"]:
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce", utc=True)

    return cleaned.sort_values("Stars", ascending=False).head(500)


def preprocess_repositories(input_path: Path, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Load, clean, and save the repository dataset."""
    dataframe = canonicalize_columns(pd.read_csv(input_path))
    original_rows = len(dataframe)

    validate_columns(dataframe, input_path)
    cleaned = clean_repositories(dataframe)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"Input file: {input_path}")
    print(f"Original rows: {original_rows}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Output file: {output_path}")


def main() -> None:
    """Run preprocessing from the command line."""
    args = parse_args()

    try:
        input_path = args.input if args.input else find_downloaded_dataset_csv()
        preprocess_repositories(input_path)
    except Exception as exc:
        print("Failed to preprocess the repositories dataset.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
