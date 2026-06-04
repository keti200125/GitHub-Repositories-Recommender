"""Streamlit demo for GitHub repository recommendations."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ollama_baseline import (
    DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL,
    OllamaUnavailableError,
    run_ollama_baseline,
)
from src.recommender import recommend_graph, recommend_hybrid, recommend_semantic


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORIES_PATH = PROJECT_ROOT / "data" / "processed" / "repositories_clean.csv"
RETRIEVAL_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "retrieval_evaluation_summary.csv"
COVERAGE_DIVERSITY_PATH = PROJECT_ROOT / "outputs" / "coverage_diversity_results.csv"
HUMAN_EVALUATION_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "human_evaluation_summary.csv"
SAMPLE_SIZE = 15
MAX_SELECTED_REPOSITORIES = 5
TOP_RECOMMENDATIONS = 5

PROFILE_COLUMNS = ["Name", "Description", "Stars", "Forks", "Issues"]
BASE_RECOMMENDATION_COLUMNS = ["Name", "Description", "Stars", "Forks", "Issues", "URL"]
RECOMMENDATION_METHODS = ["semantic", "graph", "hybrid"]
METHOD_SCORE_COLUMNS = {
    "semantic": ["semantic_similarity"],
    "graph": ["graph_similarity"],
    "hybrid": ["semantic_similarity", "graph_similarity", "popularity_score", "final_score"],
}
METHOD_SORT_COLUMNS = {
    "semantic": "semantic_similarity",
    "graph": "graph_similarity",
    "hybrid": "final_score",
}
METHOD_LABELS = {
    "semantic": "Semantic",
    "graph": "Graph",
    "hybrid": "Hybrid",
}
METHOD_EXPLANATIONS = {
    "semantic": "Semantic recommendations compare repository descriptions using text embeddings. Repositories with similar descriptions are ranked higher.",
    "graph": "Graph recommendations use node2vec graph embeddings. They focus on repositories that are close in the repository relationship graph.",
    "hybrid": "Hybrid recommendations combine semantic similarity, graph similarity, and repository popularity into one ranking.",
}
METRIC_CHARTS = [
    ("Precision@5", PROJECT_ROOT / "outputs" / "precision_at_5.png"),
    ("Recall@5", PROJECT_ROOT / "outputs" / "recall_at_5.png"),
    ("MRR", PROJECT_ROOT / "outputs" / "mrr.png"),
]
GRAPH_DEMO_IMAGES = [
    ("Full Graph", PROJECT_ROOT / "outputs" / "graph_overview.png"),
    ("Communities", PROJECT_ROOT / "outputs" / "graph_communities.png"),
    ("TensorFlow Neighborhood", PROJECT_ROOT / "outputs" / "graph_neighborhood_tensorflow.png"),
]
EVALUATION_DISPLAY_COLUMNS = {
    "method": "Method",
    "precision_at_5": "Precision@5",
    "recall_at_5": "Recall@5",
    "hit_rate_at_5": "HitRate@5",
    "mrr": "MRR",
}
COVERAGE_DIVERSITY_DISPLAY_COLUMNS = {
    "method": "Method",
    "coverage": "Coverage",
    "unique_recommended_count": "Unique Recommended",
    "total_repository_count": "Total Repositories",
    "average_diversity": "Average Diversity",
}
HUMAN_EVALUATION_DISPLAY_COLUMNS = {
    "method": "Method",
    "average_human_score": "Average Human Score",
    "ndcg_at_5": "NDCG@5",
}


@st.cache_data
def load_repositories(path: Path = REPOSITORIES_PATH) -> pd.DataFrame:
    """Load cleaned repository metadata."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned repository dataset not found: {path}. "
            "Run python3 src/data_preprocessing.py first."
        )

    return pd.read_csv(path)


@st.cache_data
def load_retrieval_summary(path: Path = RETRIEVAL_SUMMARY_PATH) -> pd.DataFrame:
    """Load saved retrieval evaluation metrics."""
    if not path.exists():
        return pd.DataFrame(columns=EVALUATION_DISPLAY_COLUMNS)

    return pd.read_csv(path)


@st.cache_data
def load_coverage_diversity_results(path: Path = COVERAGE_DIVERSITY_PATH) -> pd.DataFrame:
    """Load saved coverage and diversity evaluation metrics."""
    if not path.exists():
        return pd.DataFrame(columns=COVERAGE_DIVERSITY_DISPLAY_COLUMNS)

    return pd.read_csv(path)


@st.cache_data
def load_human_evaluation_summary(path: Path = HUMAN_EVALUATION_SUMMARY_PATH) -> pd.DataFrame:
    """Load saved human relevance evaluation metrics."""
    if not path.exists():
        return pd.DataFrame(columns=HUMAN_EVALUATION_DISPLAY_COLUMNS)

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_seed_recommendations(repo_name: str, method: str, top_k: int) -> pd.DataFrame:
    """Generate recommendations for one seed repository."""
    if method == "semantic":
        return recommend_semantic(repo_name, top_k)
    if method == "graph":
        return recommend_graph(repo_name, top_k)
    if method == "hybrid":
        return recommend_hybrid(repo_name, top_k)

    raise ValueError(f"Unsupported recommendation method: {method}")


@st.cache_data(show_spinner=False)
def get_ollama_baseline(selected_repositories: tuple[str, ...]):
    """Run the Ollama baseline for a selected profile only when requested."""
    return run_ollama_baseline(
        list(selected_repositories),
        model=OLLAMA_DEFAULT_MODEL,
    )


def sample_repository_names(repositories: pd.DataFrame) -> list[str]:
    """Randomly sample repository names for the selector."""
    repository_names = repositories["Name"].dropna().astype(str)
    sample_count = min(SAMPLE_SIZE, len(repository_names))
    return repository_names.sample(n=sample_count).sort_values().tolist()


def initialize_repository_options(repositories: pd.DataFrame) -> None:
    """Initialize repository selector and recommendation state."""
    if "repository_options" not in st.session_state:
        st.session_state.repository_options = sample_repository_names(repositories)
    if "selected_repositories" not in st.session_state:
        st.session_state.selected_repositories = []


def reset_repository_options(repositories: pd.DataFrame) -> None:
    """Refresh the random repository selector options."""
    st.session_state.repository_options = sample_repository_names(repositories)
    st.session_state.selected_repositories = []
    st.session_state.pop("recommendations", None)
    st.session_state.pop("recommendation_profile", None)
    st.session_state.pop("recommendation_elapsed_times", None)


def selected_repository_details(repositories: pd.DataFrame, selected_repositories: list[str]) -> pd.DataFrame:
    """Return metadata for selected repositories in selection order."""
    selected = repositories[repositories["Name"].astype(str).isin(selected_repositories)].copy()
    selected["_selection_order"] = selected["Name"].astype(str).map(
        {name: index for index, name in enumerate(selected_repositories)}
    )
    return selected.sort_values("_selection_order").loc[:, PROFILE_COLUMNS]


def repository_url_lookup(repositories: pd.DataFrame) -> pd.DataFrame:
    """Return repository URLs keyed by repository name."""
    if "URL" not in repositories.columns:
        return pd.DataFrame(columns=["Name", "URL"])

    return repositories.loc[:, ["Name", "URL"]].drop_duplicates(subset=["Name"])


def build_profile_recommendations(
    selected_repositories: tuple[str, ...],
    method: str,
    repositories: pd.DataFrame,
    top_k: int = TOP_RECOMMENDATIONS,
) -> pd.DataFrame:
    """Aggregate recommendations across selected repositories."""
    candidate_pool_size = max(top_k, len(repositories) - 1)
    selected_names = {name.casefold() for name in selected_repositories}
    recommendation_frames = []

    for repo_name in selected_repositories:
        recommendations = get_seed_recommendations(repo_name, method, candidate_pool_size)
        recommendation_names = recommendations["Name"].astype(str).str.casefold()
        recommendations = recommendations[~recommendation_names.isin(selected_names)].copy()
        recommendation_frames.append(recommendations)

    if not recommendation_frames:
        return pd.DataFrame(columns=[*BASE_RECOMMENDATION_COLUMNS, *METHOD_SCORE_COLUMNS[method]])

    combined = pd.concat(recommendation_frames, ignore_index=True)
    aggregation = {
        "Description": "first",
        "Stars": "first",
        "Forks": "first",
        "Issues": "first",
        **{column: "mean" for column in METHOD_SCORE_COLUMNS[method]},
    }

    profile_recommendations = combined.groupby("Name", as_index=False).agg(aggregation)
    profile_recommendations = profile_recommendations.merge(
        repository_url_lookup(repositories),
        on="Name",
        how="left",
    )
    profile_recommendations = profile_recommendations.sort_values(
        METHOD_SORT_COLUMNS[method],
        ascending=False,
    )

    columns = [*BASE_RECOMMENDATION_COLUMNS, *METHOD_SCORE_COLUMNS[method]]
    return profile_recommendations.loc[:, columns].head(top_k).reset_index(drop=True)


def build_all_profile_recommendations(
    selected_repositories: tuple[str, ...],
    repositories: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Generate recommendations for every method at once."""
    recommendations_by_method = {}
    elapsed_times = {}

    for method in RECOMMENDATION_METHODS:
        start_time = time.perf_counter()
        recommendations_by_method[method] = build_profile_recommendations(
            selected_repositories,
            method,
            repositories,
        )
        elapsed_times[method] = time.perf_counter() - start_time

    return recommendations_by_method, elapsed_times


def format_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Round recommendation score columns for display."""
    display_table = recommendations.copy()
    for column in ["semantic_similarity", "graph_similarity", "popularity_score", "final_score"]:
        if column in display_table.columns:
            display_table[column] = pd.to_numeric(display_table[column], errors="coerce").round(4)

    return display_table


def format_retrieval_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Format retrieval metrics for display."""
    display_table = summary.copy()
    if "mode" in display_table.columns:
        display_table = display_table[display_table["mode"] == "leave_one_out"].copy()

    display_columns = [column for column in EVALUATION_DISPLAY_COLUMNS if column in display_table.columns]
    display_table = display_table.loc[:, display_columns]

    if "method" in display_table.columns:
        display_table["method"] = display_table["method"].astype(str).str.title()

    for column in ["precision_at_5", "recall_at_5", "hit_rate_at_5", "mrr"]:
        if column in display_table.columns:
            display_table[column] = pd.to_numeric(display_table[column], errors="coerce").round(4)

    return display_table.rename(columns=EVALUATION_DISPLAY_COLUMNS)


def format_coverage_diversity_results(results: pd.DataFrame) -> pd.DataFrame:
    """Format coverage and diversity metrics for display."""
    display_table = results.copy()
    display_columns = [
        column for column in COVERAGE_DIVERSITY_DISPLAY_COLUMNS if column in display_table.columns
    ]
    display_table = display_table.loc[:, display_columns]

    if "method" in display_table.columns:
        display_table["method"] = display_table["method"].astype(str).str.title()

    for column in ["coverage", "average_diversity"]:
        if column in display_table.columns:
            display_table[column] = pd.to_numeric(display_table[column], errors="coerce").round(4)

    return display_table.rename(columns=COVERAGE_DIVERSITY_DISPLAY_COLUMNS)


def format_human_evaluation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Format human relevance metrics for display."""
    display_table = summary.copy()
    display_columns = [
        column for column in HUMAN_EVALUATION_DISPLAY_COLUMNS if column in display_table.columns
    ]
    display_table = display_table.loc[:, display_columns]

    if "method" in display_table.columns:
        display_table["method"] = display_table["method"].astype(str).str.title()

    for column in ["average_human_score", "ndcg_at_5"]:
        if column in display_table.columns:
            display_table[column] = pd.to_numeric(display_table[column], errors="coerce").round(4)

    return display_table.rename(columns=HUMAN_EVALUATION_DISPLAY_COLUMNS)


def recommendation_column_config() -> dict[str, object]:
    """Return Streamlit column config for readable recommendation tables."""
    return {
        "URL": st.column_config.LinkColumn("URL", display_text="Open"),
        "Description": st.column_config.TextColumn("Description", width="large"),
    }


def render_sidebar() -> bool:
    """Render sidebar controls and return selected settings."""
    with st.sidebar:
        st.header("Settings")
        st.metric("Number of recommendations", TOP_RECOMMENDATIONS)
        compare_with_ollama = st.checkbox("Compare with local LLM (Ollama)")

    return compare_with_ollama


def render_user_profile(repositories: pd.DataFrame) -> list[str]:
    """Render the repository profile selector."""
    st.header("1. User Profile")
    st.write("Choose repositories that represent the user's interests.")

    if st.button("Generate New Repository Set", type="secondary"):
        reset_repository_options(repositories)

    selected_repositories = st.multiselect(
        "Random repository sample",
        options=st.session_state.repository_options,
        key="selected_repositories",
        max_selections=MAX_SELECTED_REPOSITORIES,
        help="Select between 1 and 5 repositories from this random sample.",
    )

    if not selected_repositories:
        st.warning("Select at least one repository to generate recommendations.")
    elif len(selected_repositories) > MAX_SELECTED_REPOSITORIES:
        st.warning("Select no more than 5 repositories.")
    else:
        st.dataframe(
            selected_repository_details(repositories, selected_repositories),
            width="stretch",
            hide_index=True,
        )

    return selected_repositories


def render_recommendations(
    selected_repositories: list[str],
    repositories: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame] | None, tuple[str, ...] | None]:
    """Render recommendation controls and results."""
    st.header("2. Recommendations")

    valid_selection = 1 <= len(selected_repositories) <= MAX_SELECTED_REPOSITORIES
    if st.button("Generate Recommendations", type="primary", disabled=not valid_selection):
        selected_profile = tuple(selected_repositories)
        try:
            with st.spinner("Generating Semantic, Graph, and Hybrid recommendations..."):
                recommendations, elapsed_times = build_all_profile_recommendations(
                    selected_profile,
                    repositories,
                )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state.recommendations = recommendations
            st.session_state.recommendation_profile = selected_profile
            st.session_state.recommendation_elapsed_times = elapsed_times
            st.success("All recommendation methods generated.")

    recommendations = st.session_state.get("recommendations")
    selected_profile = st.session_state.get("recommendation_profile")

    if recommendations is None:
        st.info("Generate recommendations to see the Top 5 results for all methods.")
        return None, None

    if selected_profile != tuple(selected_repositories):
        st.info("The selected repositories changed. Generate recommendations again to refresh the results.")
        return None, None

    elapsed_times = st.session_state.get("recommendation_elapsed_times", {})
    tabs = st.tabs([METHOD_LABELS[method] for method in RECOMMENDATION_METHODS])
    for tab, method in zip(tabs, RECOMMENDATION_METHODS, strict=True):
        with tab:
            st.dataframe(
                format_recommendations(recommendations[method]),
                width="stretch",
                hide_index=True,
                column_config=recommendation_column_config(),
            )
            elapsed_time = elapsed_times.get(method, 0.0)
            st.caption(f"Top {TOP_RECOMMENDATIONS} {METHOD_LABELS[method]} results generated in {elapsed_time:.2f}s.")

    return recommendations, selected_profile


def render_explanation() -> None:
    """Render simple explanations for all recommendation methods."""
    st.header("3. Explanation")
    for method in RECOMMENDATION_METHODS:
        st.markdown(f"**{METHOD_LABELS[method]}:** {METHOD_EXPLANATIONS[method]}")


def render_evaluation_metrics() -> None:
    """Render saved retrieval evaluation metrics and charts."""
    st.header("5. Evaluation Metrics")

    summary = load_retrieval_summary()
    coverage_diversity = load_coverage_diversity_results()
    human_summary = load_human_evaluation_summary()
    if summary.empty and coverage_diversity.empty and human_summary.empty:
        st.info("No evaluation summary found yet. Run `python src/evaluate_retrieval.py` to generate it.")
        return

    if not summary.empty:
        st.subheader("Retrieval Accuracy")
        st.dataframe(
            format_retrieval_summary(summary),
            width="stretch",
            hide_index=True,
        )

        available_charts = [(label, path) for label, path in METRIC_CHARTS if path.exists()]
        if available_charts:
            columns = st.columns(len(available_charts))
            for column, (label, path) in zip(columns, available_charts, strict=True):
                with column:
                    st.image(path.as_posix(), caption=label)

    if not coverage_diversity.empty:
        st.subheader("Coverage and Diversity")
        st.write(
            "Coverage shows how much of the repository catalog each method explores. "
            "Diversity shows whether the Top 5 recommendations are varied or too similar to each other."
        )
        st.dataframe(
            format_coverage_diversity_results(coverage_diversity),
            width="stretch",
            hide_index=True,
        )

    if not human_summary.empty:
        st.subheader("Human Relevance Evaluation")
        st.write(
            "Human relevance uses manual ratings from 0 to 3. "
            "NDCG@5 rewards methods that place highly relevant repositories near the top."
        )
        st.dataframe(
            format_human_evaluation_summary(human_summary),
            width="stretch",
            hide_index=True,
        )


def render_ollama_comparison(compare_with_ollama: bool, selected_profile: tuple[str, ...] | None) -> None:
    """Render optional local LLM comparison."""
    if not compare_with_ollama:
        return

    st.header("4. Optional LLM Comparison")
    if not selected_profile:
        st.info("Generate recommendations first to compare them with Ollama.")
        return

    st.write(f"Local model: `{OLLAMA_DEFAULT_MODEL}`")
    st.write("Selected repositories: " + ", ".join(selected_profile))

    try:
        with st.spinner("Asking local Ollama model..."):
            result = get_ollama_baseline(selected_profile)
    except OllamaUnavailableError:
        st.warning("Ollama is not running. Start it with `ollama serve`, then try again.")
    except Exception as exc:
        if exc.__class__.__name__ == "OllamaTimeoutError":
            st.warning(str(exc))
        else:
            st.error(str(exc))
    else:
        if result.removed_selected_repositories:
            removed = ", ".join(result.removed_selected_repositories)
            st.warning(f"Removed selected repositories from Ollama response: {removed}")
        if result.answer:
            st.markdown(result.answer)
        else:
            st.warning("Ollama did not return any valid recommendations after filtering selected repositories.")


def render_demo_notes() -> None:
    """Render concise notes that help present the demo clearly."""
    st.header("Demo Highlights")

    st.markdown(
        """
        **Best flow:** select 1 to 5 repositories, generate recommendations, then compare the Semantic,
        Graph, and Hybrid tabs side by side.

        **Main message:** Semantic is strongest when the evaluation is based on text similarity, Graph
        captures structural relationships between repositories, and Hybrid combines semantic similarity,
        graph similarity, and popularity into one balanced ranking.

        **Evaluation caveat:** the dataset has no real user ratings or clicks, so the automatic metrics use
        back-translated queries and leave-one-out semantic neighbors as an approximate relevance signal.
        Human ratings and diversity/coverage are included to give a broader quality view.

        **Good demo examples:** `tensorflow`, `react`, `vue`, `kubernetes`, `freeCodeCamp`, and `django`
        usually produce easy-to-explain recommendation groups.
        """
    )

    available_graph_images = [(label, path) for label, path in GRAPH_DEMO_IMAGES if path.exists()]
    if available_graph_images:
        st.subheader("Repository Graph Examples")
        graph_tabs = st.tabs([label for label, _ in available_graph_images])
        for tab, (label, path) in zip(graph_tabs, available_graph_images, strict=True):
            with tab:
                st.image(path.as_posix(), caption=label)


def main() -> None:
    """Render the Streamlit demo."""
    st.set_page_config(page_title="GitHub Repositories Recommender", layout="wide")

    st.title("GitHub Repositories Recommender")
    st.write(
        "This system recommends GitHub repositories using semantic similarity, "
        "graph embeddings, and hybrid ranking."
    )

    try:
        repositories = load_repositories()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    initialize_repository_options(repositories)
    compare_with_ollama = render_sidebar()

    selected_repositories = render_user_profile(repositories)
    _, selected_profile = render_recommendations(selected_repositories, repositories)
    render_explanation()
    render_ollama_comparison(compare_with_ollama, selected_profile)
    render_evaluation_metrics()
    render_demo_notes()


if __name__ == "__main__":
    main()
