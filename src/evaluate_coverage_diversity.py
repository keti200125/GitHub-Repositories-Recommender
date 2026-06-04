"""Evaluate catalog coverage and recommendation diversity."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .ollama_baseline import recommend_ollama
    from .recommender import (
        load_embeddings,
        load_repositories,
        recommend_graph,
        recommend_hybrid,
        recommend_semantic,
        validate_semantic_inputs,
    )
except ImportError:
    from ollama_baseline import recommend_ollama
    from recommender import (
        load_embeddings,
        load_repositories,
        recommend_graph,
        recommend_hybrid,
        recommend_semantic,
        validate_semantic_inputs,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "coverage_diversity_results.csv"
TOP_K = 5
QUERY_REPOSITORIES = [
    "tensorflow",
    "react",
    "vue",
    "flutter",
    "kubernetes",
    "freeCodeCamp",
    "bootstrap",
    "transformers",
    "django",
    "linux",
]
METHODS: dict[str, Callable[[str, int], pd.DataFrame]] = {
    "semantic": recommend_semantic,
    "graph": recommend_graph,
    "hybrid": recommend_hybrid,
}
OUTPUT_COLUMNS = [
    "method",
    "coverage",
    "unique_recommended_count",
    "total_repository_count",
    "average_diversity",
]


def existing_query_repositories(repositories: pd.DataFrame) -> list[str]:
    """Return requested query repositories that exist in the catalog."""
    repository_names = repositories["Name"].astype(str)
    available_names = {name.casefold(): name for name in repository_names}
    return [
        available_names[query_repository.casefold()]
        for query_repository in QUERY_REPOSITORIES
        if query_repository.casefold() in available_names
    ]


def repository_index_lookup(repositories: pd.DataFrame) -> dict[str, int]:
    """Build a case-insensitive repository-name to row-index mapping."""
    return {
        str(row["Name"]).casefold(): int(index)
        for index, row in repositories.iterrows()
    }


def recommend_ollama_for_query(query_repository: str, top_k: int = TOP_K) -> pd.DataFrame:
    """Return Ollama recommendations for one query repository."""
    # Ollama is evaluated as an LLM-based recommendation baseline using the
    # same coverage and diversity framework as the other recommendation methods.
    return recommend_ollama([query_repository], top_k=top_k)


def intra_list_diversity(
    recommendations: pd.DataFrame,
    embeddings: np.ndarray,
    index_by_name: dict[str, int],
) -> float:
    """Compute 1 - average pairwise cosine similarity for one Top-K list."""
    recommended_indices = [
        index_by_name[str(repository_name).casefold()]
        for repository_name in recommendations["Name"]
        if str(repository_name).casefold() in index_by_name
    ]
    if len(recommended_indices) < 2:
        return 0.0

    recommendation_embeddings = embeddings[recommended_indices]
    similarities = cosine_similarity(recommendation_embeddings)
    upper_triangle_indices = np.triu_indices(len(recommended_indices), k=1)
    average_similarity = float(similarities[upper_triangle_indices].mean())
    return 1.0 - average_similarity


def evaluate_coverage_diversity() -> pd.DataFrame:
    """Evaluate catalog coverage and average intra-list diversity per method."""
    repositories = load_repositories()
    embeddings = load_embeddings()
    validate_semantic_inputs(repositories, embeddings)

    query_repositories = existing_query_repositories(repositories)
    if not query_repositories:
        raise ValueError("None of the requested query repositories exist in the cleaned dataset.")

    total_repository_count = len(repositories)
    index_by_name = repository_index_lookup(repositories)
    methods: dict[str, Callable[[str, int], pd.DataFrame]] = {
        **METHODS,
        "ollama": recommend_ollama_for_query,
    }
    rows = []

    for method, recommend in methods.items():
        unique_recommendations: set[str] = set()
        diversity_scores = []

        for query_repository in query_repositories:
            recommendations = recommend(query_repository, TOP_K)
            recommendation_names = recommendations["Name"].astype(str)
            unique_recommendations.update(name.casefold() for name in recommendation_names)
            diversity_scores.append(intra_list_diversity(recommendations, embeddings, index_by_name))

        # Coverage shows how much of the repository catalog the method explores
        # across all recommendation lists.
        coverage = len(unique_recommendations) / total_repository_count

        # Diversity shows whether each Top-5 list is varied or too similar to itself.
        # Higher intra-list diversity means recommendations are less redundant.
        average_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0

        rows.append(
            {
                "method": method,
                "coverage": round(coverage, 4),
                "unique_recommended_count": len(unique_recommendations),
                "total_repository_count": total_repository_count,
                "average_diversity": round(average_diversity, 4),
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_results(results: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Save coverage and diversity metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)


def main() -> None:
    """Run coverage and diversity evaluation."""
    try:
        results = evaluate_coverage_diversity()
        save_results(results)
    except Exception as exc:
        print("Failed to evaluate coverage and diversity.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(results.to_string(index=False))
    print(f"\nOutput path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
