"""Create a manual relevance-rating sheet for recommendation quality.

The evaluator should manually fill human_relevance with:
0 = irrelevant
1 = weakly relevant
2 = relevant
3 = highly relevant
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from .ollama_baseline import recommend_ollama
    from .recommender import load_repositories, recommend_graph, recommend_hybrid, recommend_semantic
except ImportError:
    from ollama_baseline import recommend_ollama
    from recommender import load_repositories, recommend_graph, recommend_hybrid, recommend_semantic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "human_eval_sheet.csv"
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


def existing_query_repositories(repositories: pd.DataFrame) -> list[str]:
    """Return requested query repositories that exist in the cleaned dataset."""
    available_names = {
        str(name).casefold(): str(name)
        for name in repositories["Name"].dropna()
    }
    return [
        available_names[query_repository.casefold()]
        for query_repository in QUERY_REPOSITORIES
        if query_repository.casefold() in available_names
    ]


def recommendation_row(query_repository: str, method: str, rank: int, row: pd.Series) -> dict[str, object]:
    """Convert one recommendation into one manually rateable row."""
    return {
        "query_repository": query_repository,
        "method": method,
        "rank": rank,
        "recommended_repository": row["Name"],
        "recommended_description": row["Description"],
        "stars": row["Stars"],
        "forks": row["Forks"],
        "issues": row["Issues"],
        "human_relevance": "",
    }


def recommend_ollama_for_query(query_repository: str, top_k: int = TOP_K) -> pd.DataFrame:
    """Return Ollama recommendations for one query repository."""
    # Ollama is an LLM-based recommendation baseline and should be rated using
    # the same 0-3 human relevance scale as the other methods.
    return recommend_ollama([query_repository], top_k=top_k)


def create_human_eval_sheet() -> pd.DataFrame:
    """Generate Top-5 recommendations for each query and method."""
    repositories = load_repositories()
    query_repositories = existing_query_repositories(repositories)
    if not query_repositories:
        raise ValueError("None of the requested query repositories exist in the cleaned dataset.")

    rows = []
    methods: dict[str, Callable[[str, int], pd.DataFrame]] = {
        **METHODS,
        "ollama": recommend_ollama_for_query,
    }
    for query_repository in query_repositories:
        for method, recommend in methods.items():
            recommendations = recommend(query_repository, TOP_K)
            for rank, (_, recommendation) in enumerate(recommendations.iterrows(), start=1):
                rows.append(recommendation_row(query_repository, method, rank, recommendation))

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_sheet(sheet: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Save the manual evaluation sheet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(output_path, index=False)


def main() -> None:
    """Create the blank human evaluation sheet."""
    try:
        sheet = create_human_eval_sheet()
        save_sheet(sheet)
    except Exception as exc:
        print("Failed to create human evaluation sheet.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Rows: {len(sheet)}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
