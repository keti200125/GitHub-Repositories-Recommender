"""Evaluate manually rated recommendation relevance scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "human_eval_sheet.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "human_evaluation_summary.csv"
DETAILS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "human_evaluation_details.csv"
VALID_RELEVANCE_VALUES = {0, 1, 2, 3}
TOP_K = 5
REQUIRED_COLUMNS = [
    "query_repository",
    "method",
    "rank",
    "recommended_repository",
    "recommended_description",
    "stars",
    "forks",
    "issues",
    "human_relevance",
]
SUMMARY_COLUMNS = [
    "method",
    "average_human_score",
    "ndcg_at_5",
]
METHOD_ORDER = ["semantic", "graph", "hybrid", "ollama"]


def load_human_eval_sheet(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the manually filled human evaluation sheet."""
    if not path.exists():
        raise FileNotFoundError(f"Human evaluation sheet not found: {path}")

    sheet = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in sheet.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return sheet


def validate_human_relevance(sheet: pd.DataFrame) -> pd.DataFrame:
    """Ensure all human relevance ratings are filled and valid."""
    missing_mask = sheet["human_relevance"].isna() | sheet["human_relevance"].astype(str).str.strip().eq("")
    if missing_mask.any():
        missing_count = int(missing_mask.sum())
        raise ValueError(f"human_relevance must be filled for every row. Missing rows: {missing_count}")

    validated = sheet.copy()
    validated["human_relevance"] = pd.to_numeric(validated["human_relevance"], errors="coerce")
    if validated["human_relevance"].isna().any():
        raise ValueError("human_relevance must contain only numeric values: 0, 1, 2, or 3.")

    validated["human_relevance"] = validated["human_relevance"].astype(int)
    invalid_values = sorted(set(validated["human_relevance"]) - VALID_RELEVANCE_VALUES)
    if invalid_values:
        invalid = ", ".join(str(value) for value in invalid_values)
        raise ValueError(f"human_relevance contains invalid values: {invalid}. Use only 0, 1, 2, or 3.")

    return validated


def dcg_at_k(relevance_scores: list[int], k: int = TOP_K) -> float:
    """Compute discounted cumulative gain at K."""
    scores = np.asarray(relevance_scores[:k], dtype=float)
    if scores.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, scores.size + 2))
    gains = (2**scores - 1) / discounts
    return float(gains.sum())


def ndcg_at_k(relevance_scores: list[int], k: int = TOP_K) -> float:
    """Compute normalized discounted cumulative gain at K."""
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)
    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(relevance_scores, k) / ideal_dcg


def add_query_ndcg(sheet: pd.DataFrame) -> pd.DataFrame:
    """Add one NDCG@5 value per query/method group."""
    details = sheet.copy()
    details["rank"] = pd.to_numeric(details["rank"], errors="coerce").astype(int)
    details = details.sort_values(["method", "query_repository", "rank"])

    ndcg_values = []
    for _, group in details.groupby(["method", "query_repository"], sort=False):
        relevance_scores = group.sort_values("rank")["human_relevance"].astype(int).tolist()
        ndcg = ndcg_at_k(relevance_scores)
        ndcg_values.extend([ndcg] * len(group))

    details["ndcg_at_5_for_query"] = ndcg_values
    return details


def summarize_human_scores(details: pd.DataFrame) -> pd.DataFrame:
    """Compute average human score and mean NDCG@5 per method."""
    # Ollama is treated as an LLM-based recommendation baseline and is scored
    # with the same human relevance and NDCG@5 framework as the other methods.
    query_scores = details.drop_duplicates(subset=["method", "query_repository"])
    summary = (
        details.groupby("method", as_index=False)["human_relevance"]
        .mean()
        .rename(columns={"human_relevance": "average_human_score"})
    )
    ndcg_summary = (
        query_scores.groupby("method", as_index=False)["ndcg_at_5_for_query"]
        .mean()
        .rename(columns={"ndcg_at_5_for_query": "ndcg_at_5"})
    )

    summary = summary.merge(ndcg_summary, on="method", how="left")
    summary["average_human_score"] = summary["average_human_score"].round(4)
    summary["ndcg_at_5"] = summary["ndcg_at_5"].round(4)
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    summary = (
        summary.assign(_method_order=summary["method"].map(method_order).fillna(len(method_order)))
        .sort_values("_method_order")
        .drop(columns="_method_order")
    )
    return summary.loc[:, SUMMARY_COLUMNS]


def save_results(details: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Save detailed and summary human evaluation results."""
    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(DETAILS_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)


def main() -> None:
    """Validate human ratings and save human evaluation metrics."""
    try:
        sheet = load_human_eval_sheet()
        validated = validate_human_relevance(sheet)
        details = add_query_ndcg(validated)
        summary = summarize_human_scores(details)
        save_results(details, summary)
    except Exception as exc:
        print("Failed to evaluate human relevance scores.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(summary.to_string(index=False))
    print(f"\nDetailed results: {DETAILS_OUTPUT_PATH}")
    print(f"Summary results: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
