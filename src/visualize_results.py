"""Create simple plots from retrieval evaluation results."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "retrieval_evaluation_summary.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
LOCAL_CACHE_DIR = PROJECT_ROOT / "outputs" / ".cache"

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PLOTS = [
    ("precision_at_5", "Precision@5 by Method", "Precision@5", OUTPUTS_DIR / "precision_at_5.png"),
    ("recall_at_5", "Recall@5 by Method", "Recall@5", OUTPUTS_DIR / "recall_at_5.png"),
    ("mrr", "MRR by Method", "MRR", OUTPUTS_DIR / "mrr.png"),
]
REQUIRED_COLUMNS = ["method", "precision_at_5", "recall_at_5", "mrr"]
METHOD_ORDER = ["semantic", "graph", "hybrid", "ollama"]
BAR_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]


def load_summary(path: Path = SUMMARY_PATH) -> pd.DataFrame:
    """Load retrieval evaluation summary metrics."""
    if not path.exists():
        raise FileNotFoundError(f"Retrieval evaluation summary not found: {path}")

    summary = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in summary.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {path}: {missing}")

    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    return (
        summary.copy()
        .assign(_method_order=summary["method"].map(method_order).fillna(len(method_order)))
        .sort_values("_method_order")
        .drop(columns="_method_order")
    )


def plot_metric(summary: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: Path) -> None:
    """Create and save one bar chart with value labels."""
    import matplotlib.pyplot as plt

    methods = summary["method"].astype(str).str.title()
    values = pd.to_numeric(summary[metric], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, values, color=BAR_COLORS[: len(methods)])

    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, float(values.max()) + 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate evaluation metric visualizations."""
    try:
        summary = load_summary()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        for metric, title, ylabel, output_path in PLOTS:
            plot_metric(summary, metric, title, ylabel, output_path)
    except Exception as exc:
        print("Failed to create evaluation result visualizations.")
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    for _, _, _, output_path in PLOTS:
        print(f"Chart output: {output_path}")


if __name__ == "__main__":
    main()
