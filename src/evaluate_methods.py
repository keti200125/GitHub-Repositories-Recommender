"""Evaluate semantic, graph, and hybrid recommendation methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .recommender import (
        load_repositories,
        recommend_graph,
        recommend_hybrid,
        recommend_semantic,
    )
except ImportError:
    from recommender import (
        load_repositories,
        recommend_graph,
        recommend_hybrid,
        recommend_semantic,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PATH = OUTPUTS_DIR / "evaluation_results.csv"
TEST_REPOSITORIES = [
    "tensorflow",
    "react",
    "kubernetes",
    "freeCodeCamp",
    "transformers",
]
TOP_K = 5
METHODS = {
    "semantic": recommend_semantic,
    "graph": recommend_graph,
    "hybrid": recommend_hybrid,
}
EVALUATION_COLUMNS = [
    "query_repository",
    "method",
    "rank",
    "recommended_repository",
    "description",
    "stars",
    "forks",
    "issues",
    "semantic_similarity",
    "graph_similarity",
    "popularity_score",
    "final_score",
]
SCORE_COLUMNS = [
    "semantic_similarity",
    "graph_similarity",
    "popularity_score",
    "final_score",
]


def repository_exists(repositories: pd.DataFrame, repo_name: str) -> bool:
    """Return whether a repository exists in the cleaned dataset."""
    repository_names = repositories["Name"].astype(str).str.casefold()
    return repository_names.eq(repo_name.casefold()).any()


def prepare_evaluation_rows(
    query_repository: str,
    method: str,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Convert one recommendation table into the shared evaluation schema."""
    table = recommendations.copy()
    table.insert(0, "rank", range(1, len(table) + 1))
    table.insert(0, "method", method)
    table.insert(0, "query_repository", query_repository)

    table = table.rename(
        columns={
            "Name": "recommended_repository",
            "Description": "description",
            "Stars": "stars",
            "Forks": "forks",
            "Issues": "issues",
        }
    )

    for column in EVALUATION_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan

    for column in SCORE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").map(
            lambda value: round(float(value), 4) if pd.notna(value) else np.nan
        )

    return table.loc[:, EVALUATION_COLUMNS]


def evaluate_methods(test_repositories: list[str] = TEST_REPOSITORIES, top_k: int = TOP_K) -> pd.DataFrame:
    """Run all recommendation methods for the configured test repositories."""
    repositories = load_repositories()
    missing_repositories = [repo for repo in test_repositories if not repository_exists(repositories, repo)]
    if missing_repositories:
        missing = ", ".join(missing_repositories)
        raise ValueError(f"Test repositories not found in cleaned dataset: {missing}")

    evaluation_frames = []
    for query_repository in test_repositories:
        for method, recommend in METHODS.items():
            recommendations = recommend(query_repository, top_k)
            evaluation_frames.append(
                prepare_evaluation_rows(query_repository, method, recommendations)
            )

    return pd.concat(evaluation_frames, ignore_index=True)


def save_evaluation_results(results: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> Path:
    """Save evaluation results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path


def print_summary(results: pd.DataFrame, output_path: Path) -> None:
    """Print a compact terminal summary."""
    print("Recommendation Methods Evaluation")
    print("=================================")
    print(f"Test repositories: {', '.join(TEST_REPOSITORIES)}")
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Top-K per method: {TOP_K}")
    print(f"Rows saved: {len(results)}")
    print(f"Output file: {output_path}")
    print("\nResult counts by method:")
    print(results.groupby("method").size().to_string())


def main() -> None:
    """Evaluate recommendation methods and save the results."""
    try:
        results = evaluate_methods()
        output_path = save_evaluation_results(results)
    except Exception as exc:
        print("Failed to evaluate recommendation methods.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print_summary(results, output_path)


if __name__ == "__main__":
    main()
