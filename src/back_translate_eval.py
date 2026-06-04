"""Back-translate paraphrase evaluation queries through Japanese."""

from __future__ import annotations

from pathlib import Path
from time import sleep

import pandas as pd
from deep_translator import GoogleTranslator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation" / "paraphrase_eval.csv"
SOURCE_LANGUAGE = "en"
PIVOT_LANGUAGE = "ja"
OUTPUT_COLUMNS = [
    "query_repository",
    "original_description",
    "query_description",
    "Stars",
    "Forks",
    "Issues",
    "URL",
]


def validate_columns(evaluation_set: pd.DataFrame) -> None:
    """Ensure the evaluation file has the columns needed for back-translation."""
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in evaluation_set.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {EVALUATION_PATH}: {missing}")


def back_translate(description: str) -> str:
    """Translate English to Japanese and back to English."""
    english_to_pivot = GoogleTranslator(source=SOURCE_LANGUAGE, target=PIVOT_LANGUAGE)
    pivot_to_english = GoogleTranslator(source=PIVOT_LANGUAGE, target=SOURCE_LANGUAGE)

    pivot_description = english_to_pivot.translate(description)
    sleep(0.2)
    return pivot_to_english.translate(pivot_description)


def add_back_translated_queries(evaluation_set: pd.DataFrame) -> pd.DataFrame:
    """Replace query descriptions with back-translated retrieval queries."""
    validate_columns(evaluation_set)
    updated = evaluation_set.loc[:, OUTPUT_COLUMNS].copy()

    back_translated_descriptions = []
    for _, row in updated.iterrows():
        original_description = str(row["original_description"])
        try:
            back_translated_description = back_translate(original_description)
        except Exception as exc:
            back_translated_description = str(row["query_description"])
            print(f"Repository: {row['query_repository']}")
            print(f"Original: {original_description}")
            print(f"Translation failed: {exc}")
            print(f"Keeping existing query description: {back_translated_description}")
            print()
            back_translated_descriptions.append(back_translated_description)
            continue

        back_translated_descriptions.append(back_translated_description)

        print(f"Repository: {row['query_repository']}")
        print(f"Original: {original_description}")
        print(f"Back-translated: {back_translated_description}")
        print()

    # The back-translated description is used as the retrieval query.
    # The expected retrieval result is the original repository in query_repository.
    updated["query_description"] = back_translated_descriptions
    return updated


def main() -> None:
    """Load the evaluation set, back-translate queries, and save it in place."""
    try:
        evaluation_set = pd.read_csv(EVALUATION_PATH)
        updated = add_back_translated_queries(evaluation_set)
        updated.to_csv(EVALUATION_PATH, index=False)
    except Exception as exc:
        print("Failed to create back-translated evaluation queries.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Saved back-translated evaluation queries to: {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
