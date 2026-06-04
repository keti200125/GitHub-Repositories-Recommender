"""Generate qualitative recommendation case studies for reports."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from .recommender import recommend_graph, recommend_hybrid, recommend_semantic
except ImportError:
    from recommender import recommend_graph, recommend_hybrid, recommend_semantic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CSV_OUTPUT_PATH = OUTPUTS_DIR / "case_studies.csv"
MARKDOWN_OUTPUT_PATH = OUTPUTS_DIR / "case_studies.md"
TOP_K = 5
QUERY_REPOSITORIES = [
    "tensorflow",
    "react",
    "vue",
    "flutter",
    "kubernetes",
]
METHODS: dict[str, Callable[[str, int], pd.DataFrame]] = {
    "Semantic": recommend_semantic,
    "Graph": recommend_graph,
    "Hybrid": recommend_hybrid,
}
OUTPUT_COLUMNS = [
    "query_repository",
    "method",
    "rank",
    "recommended_repository",
    "description",
    "stars",
    "forks",
    "issues",
]


def normalize_recommendation_row(query_repository: str, method: str, rank: int, row: pd.Series) -> dict[str, object]:
    """Convert one recommendation row into the case-study output schema."""
    return {
        "query_repository": query_repository,
        "method": method.lower(),
        "rank": rank,
        "recommended_repository": row["Name"],
        "description": row["Description"],
        "stars": row["Stars"],
        "forks": row["Forks"],
        "issues": row["Issues"],
    }


def generate_case_studies() -> pd.DataFrame:
    """Generate Top-5 recommendations for each query repository and method."""
    rows = []

    for query_repository in QUERY_REPOSITORIES:
        for method_name, recommend in METHODS.items():
            recommendations = recommend(query_repository, TOP_K)

            for rank, (_, recommendation) in enumerate(recommendations.iterrows(), start=1):
                rows.append(normalize_recommendation_row(query_repository, method_name, rank, recommendation))

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def format_markdown(results: pd.DataFrame) -> str:
    """Format case-study recommendations as a Markdown report section."""
    lines = []

    for query_repository in QUERY_REPOSITORIES:
        lines.append(f"# {query_repository}")
        lines.append("")

        query_results = results[results["query_repository"] == query_repository]
        for method_name in METHODS:
            method_results = query_results[query_results["method"] == method_name.lower()]

            lines.append(f"## {method_name}")
            for _, row in method_results.iterrows():
                lines.append(
                    f"{int(row['rank'])}. {row['recommended_repository']} "
                    f"(stars: {row['stars']}, forks: {row['forks']}, issues: {row['issues']}) - "
                    f"{row['description']}"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_case_studies(results: pd.DataFrame) -> None:
    """Save case studies as CSV and Markdown."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(CSV_OUTPUT_PATH, index=False)
    MARKDOWN_OUTPUT_PATH.write_text(format_markdown(results), encoding="utf-8")


def main() -> None:
    """Generate and save qualitative recommendation case studies."""
    try:
        results = generate_case_studies()
        save_case_studies(results)
    except Exception as exc:
        print("Failed to generate qualitative recommendation case studies.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"CSV output: {CSV_OUTPUT_PATH}")
    print(f"Markdown output: {MARKDOWN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
