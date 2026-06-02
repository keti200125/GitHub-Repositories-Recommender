"""Generate Node2Vec embeddings from the repository similarity graph."""

from __future__ import annotations

import contextlib
import io
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from node2vec import Node2Vec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_PATH = PROCESSED_DATA_DIR / "repository_graph.gpickle"
NODE2VEC_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "node2vec_embeddings.pkl"

EMBEDDING_DIMENSIONS = 128
WALK_LENGTH = 20
NUM_WALKS = 100
WORKERS = 4
SEED = 42
WINDOW = 10
MIN_COUNT = 1
BATCH_WORDS = 4
EDGE_WEIGHT_KEY = "similarity_score"


def load_graph(path: Path = GRAPH_PATH) -> nx.Graph:
    """Load the repository graph from a pickle-backed gpickle file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Repository graph file not found: {path}. "
            "Run python src/graph_builder.py first."
        )

    with path.open("rb") as file:
        graph = pickle.load(file)

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected a NetworkX graph, got {type(graph).__name__}.")

    if graph.number_of_nodes() == 0:
        raise ValueError("Repository graph has no nodes.")

    return graph


def train_node2vec(graph: nx.Graph) -> Any:
    """Train a Node2Vec model on the repository graph."""
    try:
        node2vec = Node2Vec(
            graph,
            dimensions=EMBEDDING_DIMENSIONS,
            walk_length=WALK_LENGTH,
            num_walks=NUM_WALKS,
            workers=WORKERS,
            seed=SEED,
            weight_key=EDGE_WEIGHT_KEY,
            quiet=True,
        )
        with contextlib.redirect_stderr(io.StringIO()):
            return node2vec.fit(
                window=WINDOW,
                min_count=MIN_COUNT,
                batch_words=BATCH_WORDS,
                workers=1,
            )
    except Exception as exc:
        raise RuntimeError(f"Node2Vec training failed: {exc}") from exc


def save_embeddings(model: Any, output_path: Path = NODE2VEC_EMBEDDINGS_PATH) -> None:
    """Save model embeddings as a repository-name-to-vector dictionary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = {
        str(node_name): np.asarray(model.wv[node_name], dtype=np.float32)
        for node_name in model.wv.key_to_index
    }

    with output_path.open("wb") as file:
        pickle.dump(embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_node2vec_embeddings(path: Path = NODE2VEC_EMBEDDINGS_PATH) -> dict[str, np.ndarray]:
    """Load saved Node2Vec embeddings from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Node2Vec embeddings file not found: {path}. "
            "Run python src/node2vec_embeddings.py first."
        )

    with path.open("rb") as file:
        embeddings = pickle.load(file)

    if not isinstance(embeddings, dict):
        raise TypeError(f"Expected embeddings dictionary, got {type(embeddings).__name__}.")

    return {
        str(node_name): np.asarray(vector, dtype=np.float32)
        for node_name, vector in embeddings.items()
    }


def main() -> None:
    """Generate and save Node2Vec embeddings from the repository graph."""
    try:
        graph = load_graph()
        model = train_node2vec(graph)
        save_embeddings(model)
        embeddings = load_node2vec_embeddings()
    except Exception as exc:
        print("Failed to generate Node2Vec embeddings.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSIONS}")
    print(f"Saved embeddings: {len(embeddings)}")
    print(f"Output path: {NODE2VEC_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
