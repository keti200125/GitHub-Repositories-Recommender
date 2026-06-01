"""Visualize a small repository similarity graph neighborhood."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "repository_graph.gpickle"
MAX_NODES = 20

# Keep matplotlib cache files out of the user's home directory and outside Git.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "outputs" / ".cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize a repository graph neighborhood.")
    parser.add_argument("--repo", required=True, help="Repository name to visualize.")
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        choices=[1, 2],
        help="Neighborhood depth to include: 1 for direct neighbors, 2 for neighbors of neighbors.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to save the PNG image.")
    return parser.parse_args()


def load_graph(path: Path = GRAPH_PATH) -> nx.Graph:
    """Load the saved repository graph."""
    if not path.exists():
        raise FileNotFoundError(
            f"Repository graph not found: {path}. Run python src/graph_builder.py first."
        )

    with path.open("rb") as file:
        return pickle.load(file)


def edge_score(graph: nx.Graph, source: str, target: str) -> float:
    """Return the semantic similarity score for an edge, or 0 for missing edges."""
    if graph.has_edge(source, target):
        return float(graph[source][target].get("similarity_score", 0.0))

    return 0.0


def select_subgraph_nodes(graph: nx.Graph, repo_name: str, depth: int) -> list[str]:
    """Select up to MAX_NODES from a repository neighborhood."""
    if repo_name not in graph:
        raise ValueError(f"Repository not found in graph: {repo_name}")

    path_lengths = nx.single_source_shortest_path_length(graph, repo_name, cutoff=depth)
    candidates = [node for node in path_lengths if node != repo_name]
    candidates.sort(
        key=lambda node: (
            path_lengths[node],
            -edge_score(graph, repo_name, node),
            node.casefold(),
        )
    )

    return [repo_name, *candidates[: MAX_NODES - 1]]


def build_subgraph(graph: nx.Graph, repo_name: str, depth: int) -> nx.Graph:
    """Extract a readable neighborhood subgraph."""
    selected_nodes = select_subgraph_nodes(graph, repo_name, depth)
    return graph.subgraph(selected_nodes).copy()


def draw_subgraph(subgraph: nx.Graph, repo_name: str, output_path: Path) -> None:
    """Draw and save the selected repository subgraph."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 9))
    positions = nx.spring_layout(subgraph, seed=42, k=0.65)

    node_sizes = [1700 if node == repo_name else 850 for node in subgraph.nodes]
    node_colors = ["#f2a65a" if node == repo_name else "#7fb3d5" for node in subgraph.nodes]

    nx.draw_networkx_nodes(
        subgraph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#2f3640",
        linewidths=1.2,
    )
    nx.draw_networkx_edges(
        subgraph,
        positions,
        width=1.1,
        alpha=0.55,
        edge_color="#566573",
    )
    nx.draw_networkx_labels(
        subgraph,
        positions,
        font_size=8,
        font_weight="bold",
    )

    edge_labels = {
        (source, target): f"{attrs.get('similarity_score', 0.0):.2f}"
        for source, target, attrs in subgraph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        subgraph,
        positions,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.5,
    )

    plt.title("Example repository similarity graph", fontsize=16, pad=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run graph visualization from the command line."""
    args = parse_args()

    try:
        graph = load_graph()
        subgraph = build_subgraph(graph, args.repo, args.depth)
        draw_subgraph(subgraph, args.repo, args.output)
    except Exception as exc:
        print("Failed to visualize repository graph.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Selected repository: {args.repo}")
    print(f"Subgraph nodes: {subgraph.number_of_nodes()}")
    print(f"Subgraph edges: {subgraph.number_of_edges()}")
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()
