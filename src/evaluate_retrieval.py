"""Evaluate retrieval with back-translated repository descriptions as queries."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .embeddings import MODEL_NAME
    from .ollama_baseline import recommend_ollama
    from .recommender import (
        compute_popularity_scores,
        find_embedding_key,
        load_embeddings,
        load_node2vec_embeddings,
        load_repositories,
        validate_semantic_inputs,
    )
except ImportError:
    from embeddings import MODEL_NAME
    from ollama_baseline import recommend_ollama
    from recommender import (
        compute_popularity_scores,
        find_embedding_key,
        load_embeddings,
        load_node2vec_embeddings,
        load_repositories,
        validate_semantic_inputs,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation" / "paraphrase_eval.csv"
DETAILS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "retrieval_evaluation_details.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "retrieval_evaluation_summary.csv"

DIRECT_RETRIEVAL_MODE = "direct_retrieval"
LEAVE_ONE_OUT_MODE = "leave_one_out"
METHODS = ["semantic", "graph", "hybrid", "ollama"]
TOP_K = 5
RELEVANCE_SET_SIZE = 10
HYBRID_WEIGHTS = {
    "semantic": 0.45,
    "graph": 0.35,
    "popularity": 0.20,
}
EVALUATION_COLUMNS = [
    "query_repository",
    "original_description",
    "query_description",
]
DIRECT_DETAIL_COLUMNS = [
    "mode",
    "query_repository",
    "method",
    "rank_of_expected",
    "hit_at_1",
    "hit_at_5",
    "reciprocal_rank",
]
DIRECT_SUMMARY_COLUMNS = [
    "method",
    "hit_rate_at_1",
    "hit_rate_at_5",
    "mrr",
]
LEAVE_ONE_OUT_DETAIL_COLUMNS = [
    "mode",
    "query_repository",
    "method",
    "relevant_repositories",
    "recommended_repositories",
    "relevant_hits",
    "precision_at_5",
    "recall_at_5",
    "hit_at_5",
    "first_relevant_rank",
    "reciprocal_rank",
]
LEAVE_ONE_OUT_SUMMARY_COLUMNS = [
    "method",
    "precision_at_5",
    "recall_at_5",
    "hit_rate_at_5",
    "mrr",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate back-translated retrieval queries.")
    parser.add_argument(
        "--mode",
        choices=[DIRECT_RETRIEVAL_MODE, LEAVE_ONE_OUT_MODE],
        default=LEAVE_ONE_OUT_MODE,
        help="Evaluation mode. Defaults to leave_one_out to avoid direct self-retrieval leakage.",
    )
    return parser.parse_args()


def load_evaluation_set(path: Path = EVALUATION_PATH) -> pd.DataFrame:
    """Load back-translated retrieval queries."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation query file not found: {path}")

    evaluation_set = pd.read_csv(path)
    missing_columns = [column for column in EVALUATION_COLUMNS if column not in evaluation_set.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return evaluation_set


def load_sentence_transformer() -> SentenceTransformer:
    """Load the same sentence-transformer model used to create repository embeddings."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                return SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                return SentenceTransformer(MODEL_NAME)


def repository_index_by_name(repositories: pd.DataFrame, repo_name: str) -> int:
    """Return the repository index for a case-insensitive repository name match."""
    repository_names = repositories["Name"].astype(str)
    matches = repository_names[repository_names.str.casefold() == repo_name.casefold()].index
    if matches.empty:
        raise ValueError(f"Expected repository not found in cleaned dataset: {repo_name}")

    return int(matches[0])


def apply_exclusions(scores: np.ndarray, excluded_indices: set[int]) -> np.ndarray:
    """Set excluded candidate scores to negative infinity."""
    ranked_scores = scores.copy()
    if excluded_indices:
        ranked_scores[list(excluded_indices)] = -np.inf
    return ranked_scores


def rank_by_scores(scores: np.ndarray, excluded_indices: set[int] | None = None) -> list[int]:
    """Return repository indices ranked by descending score."""
    excluded_indices = excluded_indices or set()
    return np.argsort(apply_exclusions(scores, excluded_indices))[::-1].tolist()


def semantic_scores(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Compute semantic similarities between a text query and repository embeddings."""
    return cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]


def graph_similarities_from_node(
    seed_index: int,
    repositories: pd.DataFrame,
    node_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    """Return graph similarities from the mapped repository node to all nodes."""
    similarities = np.full(len(repositories), -np.inf)
    seed_name = str(repositories.iloc[seed_index]["Name"])
    seed_key = find_embedding_key(node_embeddings, seed_name)
    if seed_key is None:
        return similarities

    seed_embedding = node_embeddings[seed_key].reshape(1, -1)
    for repository_index, row in repositories.iterrows():
        candidate_key = find_embedding_key(node_embeddings, str(row["Name"]))
        if candidate_key is None:
            continue

        candidate_embedding = node_embeddings[candidate_key].reshape(1, -1)
        similarities[repository_index] = float(cosine_similarity(seed_embedding, candidate_embedding)[0, 0])

    return similarities


def semantic_neighbor_set(expected_index: int, embeddings: np.ndarray, size: int = RELEVANCE_SET_SIZE) -> list[int]:
    """Return the original repository's top semantic neighbors as relevance labels."""
    similarities = cosine_similarity(embeddings[expected_index].reshape(1, -1), embeddings)[0]
    return rank_by_scores(similarities, {expected_index})[:size]


def map_query_to_repository_node(semantic_similarities: np.ndarray, excluded_indices: set[int]) -> int:
    """Map a text query to the closest available repository node."""
    ranking = rank_by_scores(semantic_similarities, excluded_indices)
    return ranking[0]


def method_rankings(
    query_embedding: np.ndarray,
    repositories: pd.DataFrame,
    embeddings: np.ndarray,
    node_embeddings: dict[str, np.ndarray],
    popularity_scores: np.ndarray,
    excluded_indices: set[int],
    include_graph_seed: bool,
) -> dict[str, list[int]]:
    """Rank repositories for semantic, graph, and hybrid text-query retrieval."""
    query_semantic_scores = semantic_scores(query_embedding, embeddings)
    mapped_node_index = map_query_to_repository_node(query_semantic_scores, excluded_indices)

    graph_scores = graph_similarities_from_node(mapped_node_index, repositories, node_embeddings)
    graph_exclusions = set(excluded_indices)
    if include_graph_seed:
        graph_scores[mapped_node_index] = np.inf
    else:
        graph_exclusions.add(mapped_node_index)

    graph_scores_for_hybrid = np.where(np.isfinite(graph_scores), graph_scores, 0.0)
    final_scores = (
        HYBRID_WEIGHTS["semantic"] * query_semantic_scores
        + HYBRID_WEIGHTS["graph"] * graph_scores_for_hybrid
        + HYBRID_WEIGHTS["popularity"] * popularity_scores
    )

    return {
        "semantic": rank_by_scores(query_semantic_scores, excluded_indices),
        "graph": rank_by_scores(graph_scores, graph_exclusions),
        "hybrid": rank_by_scores(final_scores, excluded_indices),
    }


def find_rank(ranking: list[int], expected_index: int) -> int | None:
    """Return the 1-based rank of the expected repository, if present."""
    try:
        return ranking.index(expected_index) + 1
    except ValueError:
        return None


def names_from_indices(repositories: pd.DataFrame, indices: list[int]) -> str:
    """Return a pipe-delimited repository-name list for CSV output."""
    return "|".join(repositories.iloc[indices]["Name"].astype(str).tolist())


def first_relevant_rank(recommendations: list[int], relevance_set: set[int]) -> int | None:
    """Return the first 1-based recommendation rank that appears in the relevance set."""
    for rank, repository_index in enumerate(recommendations, start=1):
        if repository_index in relevance_set:
            return rank

    return None


def ollama_recommendation_indices(
    query_repository: str,
    repositories: pd.DataFrame,
    excluded_indices: set[int],
) -> list[int]:
    """Return Ollama baseline recommendations mapped to repository indices."""
    # Ollama serves as an LLM-based recommendation baseline and is evaluated
    # using the same ranking framework as the other approaches.
    recommendations = recommend_ollama([query_repository], top_k=TOP_K)
    repository_names = repositories["Name"].astype(str)
    index_by_name = {name.casefold(): int(index) for index, name in repository_names.items()}

    recommendation_indices = []
    for recommendation_name in recommendations["Name"].astype(str):
        index = index_by_name.get(recommendation_name.casefold())
        if index is None or index in excluded_indices:
            continue
        recommendation_indices.append(index)

    return recommendation_indices[:TOP_K]


def evaluate_direct_retrieval(
    repositories: pd.DataFrame,
    embeddings: np.ndarray,
    node_embeddings: dict[str, np.ndarray],
    popularity_scores: np.ndarray,
    evaluation_set: pd.DataFrame,
    query_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check whether a back-translated query retrieves the original repository."""
    # direct_retrieval checks whether a paraphrased query retrieves the original
    # repository, so the original repository remains in the candidate pool.
    detail_rows = []

    for row_index, row in evaluation_set.iterrows():
        query_repository = str(row["query_repository"])
        expected_index = repository_index_by_name(repositories, query_repository)
        rankings = method_rankings(
            query_embeddings[row_index],
            repositories,
            embeddings,
            node_embeddings,
            popularity_scores,
            excluded_indices=set(),
            include_graph_seed=True,
        )
        rankings["ollama"] = ollama_recommendation_indices(query_repository, repositories, set())

        for method in METHODS:
            rank_of_expected = find_rank(rankings[method], expected_index)
            reciprocal_rank = 0.0 if rank_of_expected is None else 1.0 / rank_of_expected
            detail_rows.append(
                {
                    "mode": DIRECT_RETRIEVAL_MODE,
                    "query_repository": query_repository,
                    "method": method,
                    "rank_of_expected": rank_of_expected,
                    "hit_at_1": int(rank_of_expected == 1),
                    "hit_at_5": int(rank_of_expected is not None and rank_of_expected <= TOP_K),
                    "reciprocal_rank": reciprocal_rank,
                }
            )

    details = pd.DataFrame(detail_rows, columns=DIRECT_DETAIL_COLUMNS)
    summary = (
        details.groupby("method", as_index=False)
        .agg(
            hit_rate_at_1=("hit_at_1", "mean"),
            hit_rate_at_5=("hit_at_5", "mean"),
            mrr=("reciprocal_rank", "mean"),
        )
        .loc[:, DIRECT_SUMMARY_COLUMNS]
    )
    return details, summary


def evaluate_leave_one_out(
    repositories: pd.DataFrame,
    embeddings: np.ndarray,
    node_embeddings: dict[str, np.ndarray],
    popularity_scores: np.ndarray,
    evaluation_set: pd.DataFrame,
    query_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate related-repository retrieval with the source repository removed."""
    # leave_one_out is stricter because the original repository is removed and
    # each method must retrieve repositories related to that original source.
    detail_rows = []

    for row_index, row in evaluation_set.iterrows():
        query_repository = str(row["query_repository"])
        expected_index = repository_index_by_name(repositories, query_repository)
        relevance_indices = semantic_neighbor_set(expected_index, embeddings)
        relevance_set = set(relevance_indices)
        rankings = method_rankings(
            query_embeddings[row_index],
            repositories,
            embeddings,
            node_embeddings,
            popularity_scores,
            excluded_indices={expected_index},
            include_graph_seed=False,
        )
        rankings["ollama"] = ollama_recommendation_indices(
            query_repository,
            repositories,
            {expected_index},
        )

        for method in METHODS:
            recommendations = rankings[method][:TOP_K]
            relevant_hits = [index for index in recommendations if index in relevance_set]
            first_rank = first_relevant_rank(recommendations, relevance_set)
            reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
            detail_rows.append(
                {
                    "mode": LEAVE_ONE_OUT_MODE,
                    "query_repository": query_repository,
                    "method": method,
                    "relevant_repositories": names_from_indices(repositories, relevance_indices),
                    "recommended_repositories": names_from_indices(repositories, recommendations),
                    "relevant_hits": len(relevant_hits),
                    "precision_at_5": len(relevant_hits) / TOP_K,
                    "recall_at_5": len(relevant_hits) / len(relevance_set),
                    "hit_at_5": int(bool(relevant_hits)),
                    "first_relevant_rank": first_rank,
                    "reciprocal_rank": reciprocal_rank,
                }
            )

    details = pd.DataFrame(detail_rows, columns=LEAVE_ONE_OUT_DETAIL_COLUMNS)
    summary = (
        details.groupby("method", as_index=False)
        .agg(
            precision_at_5=("precision_at_5", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            hit_rate_at_5=("hit_at_5", "mean"),
            mrr=("reciprocal_rank", "mean"),
        )
        .loc[:, LEAVE_ONE_OUT_SUMMARY_COLUMNS]
    )
    return details, summary


def evaluate_retrieval(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate retrieval in the selected mode."""
    repositories = load_repositories()
    embeddings = load_embeddings()
    node_embeddings = load_node2vec_embeddings()
    evaluation_set = load_evaluation_set()
    validate_semantic_inputs(repositories, embeddings)

    model = load_sentence_transformer()
    popularity_scores = compute_popularity_scores(repositories)
    query_embeddings = model.encode(
        evaluation_set["query_description"].astype(str).tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if mode == DIRECT_RETRIEVAL_MODE:
        return evaluate_direct_retrieval(
            repositories,
            embeddings,
            node_embeddings,
            popularity_scores,
            evaluation_set,
            query_embeddings,
        )

    return evaluate_leave_one_out(
        repositories,
        embeddings,
        node_embeddings,
        popularity_scores,
        evaluation_set,
        query_embeddings,
    )


def order_methods(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort methods in the project convention order."""
    method_order = {method: index for index, method in enumerate(METHODS)}
    return (
        summary.assign(_method_order=summary["method"].map(method_order))
        .sort_values("_method_order")
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def save_results(details: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Save detailed and summary retrieval evaluation results."""
    DETAILS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(DETAILS_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)


def main() -> None:
    """Run retrieval evaluation and print the summary table."""
    args = parse_args()

    try:
        details, summary = evaluate_retrieval(args.mode)
        summary = order_methods(summary)
        metric_columns = [column for column in summary.columns if column not in {"mode", "method"}]
        summary[metric_columns] = summary[metric_columns].round(4)
        if "reciprocal_rank" in details.columns:
            details["reciprocal_rank"] = details["reciprocal_rank"].round(4)
        save_results(details, summary)
    except Exception as exc:
        print("Failed to run retrieval evaluation.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
