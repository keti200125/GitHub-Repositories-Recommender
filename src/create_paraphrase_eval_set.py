"""Create a small English-only repository set for paraphrase evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect


DetectorFactory.seed = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "repositories_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "paraphrase_eval.csv"
EVALUATION_SET_SIZE = 20
MIN_DESCRIPTION_LENGTH = 20

REQUIRED_COLUMNS = [
    "Name",
    "Description",
    "Stars",
    "Forks",
    "Issues",
    "URL",
]

OUTPUT_COLUMNS = [
    "query_repository",
    "original_description",
    "query_description",
    "Stars",
    "Forks",
    "Issues",
    "URL",
]


def is_english_description(description: str) -> bool:
    """Return whether langdetect classifies the description as English."""
    try:
        return detect(description) == "en"
    except LangDetectException:
        return False


def validate_columns(repositories: pd.DataFrame) -> None:
    """Ensure the cleaned repository file has the columns needed for evaluation."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in repositories.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {INPUT_PATH}: {missing}")


def create_paraphrase_eval_set(repositories: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build an English-only evaluation set sorted by repository stars."""
    validate_columns(repositories)

    candidates = repositories.loc[:, REQUIRED_COLUMNS].copy()
    candidates["Description"] = candidates["Description"].astype("string").str.strip()
    candidates = candidates.dropna(subset=["Description"])
    candidates = candidates[candidates["Description"].str.len() >= MIN_DESCRIPTION_LENGTH]

    for column in ["Stars", "Forks", "Issues"]:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0).astype("int64")

    candidates = candidates.sort_values("Stars", ascending=False)
    english_repositories = candidates[candidates["Description"].map(is_english_description)].copy()

    evaluation_set = english_repositories.head(EVALUATION_SET_SIZE).rename(
        columns={
            "Name": "query_repository",
            "Description": "original_description",
        }
    )

    # For now, the query text is identical to the original description.
    # Later, this column will be replaced with a paraphrased or back-translated
    # version to test whether the recommender retrieves the source repository.
    evaluation_set["query_description"] = evaluation_set["original_description"]

    evaluation_set = evaluation_set.loc[:, OUTPUT_COLUMNS]
    return evaluation_set, len(english_repositories)


def main() -> None:
    """Load cleaned repositories and save the paraphrase evaluation set."""
    try:
        repositories = pd.read_csv(INPUT_PATH)
        evaluation_set, english_count = create_paraphrase_eval_set(repositories)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        evaluation_set.to_csv(OUTPUT_PATH, index=False)
    except Exception as exc:
        print("Failed to create paraphrase evaluation set.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Total repositories loaded: {len(repositories)}")
    print(f"English repositories found: {english_count}")
    print(f"Final evaluation set size: {len(evaluation_set)}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
