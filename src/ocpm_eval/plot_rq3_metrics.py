"""Compare RQ3 model metrics using consistent colors across all charts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FILES = {
    "XGBoost": Path("data/results/rq3_results_xgboost_full.csv"),
    "GNN": Path("data/results/rq3_results_gnn_full.csv"),
    "Random Forest": Path("data/results/rq3_results_random_forest_full.csv"),
    "LSTM": Path("data/results/rq3_results_lstm_full.csv"),
    "LSTM Torch": Path("data/results/rq3_results_lstm_torch_full.csv"),
    
}

# Fixed across every figure so each model is immediately recognizable.
MODEL_COLORS = {
    "XGBoost": "#E76F51",
    "GNN": "#2A9D8F",
    "Random Forest": "#457B9D",
    "LSTM": "#9B5DE5",
    "LSTM Torch": "#E9C46A",
    
}

REQUIRED_COLUMNS = {"log", "task", "metric_name", "metric_mean", "metric_sd"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one RQ3 comparison chart for each metric."
    )
    parser.add_argument("--xgboost", type=Path, default=DEFAULT_FILES["XGBoost"])
    parser.add_argument("--gnn", type=Path, default=DEFAULT_FILES["GNN"])
    
    parser.add_argument(
        "--random-forest", type=Path, default=DEFAULT_FILES["Random Forest"]
    )
    parser.add_argument("--lstm", type=Path, default=DEFAULT_FILES["LSTM"])
    parser.add_argument(
        "--lstm-torch", type=Path, default=DEFAULT_FILES["LSTM Torch"]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/plots/rq3_metrics"),
        help="Directory where PNG charts are saved.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_results(files: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found for {model}: {path}")
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        frame = frame.copy()
        frame["model"] = model
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    results["metric_mean"] = pd.to_numeric(results["metric_mean"], errors="coerce")
    results["metric_sd"] = pd.to_numeric(results["metric_sd"], errors="coerce")
    results = results.dropna(subset=["metric_name", "metric_mean"])
    if results.empty:
        raise ValueError("The CSV files contain no valid metric values.")
    return results


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def plot_metric(data: pd.DataFrame, metric: str, output_dir: Path, dpi: int) -> Path:
    metric_data = data[data["metric_name"] == metric].copy()
    metric_data["case"] = (
        metric_data["log"].astype(str) + " | " + metric_data["task"].astype(str)
    )
    cases = list(dict.fromkeys(metric_data["case"]))
    models = list(DEFAULT_FILES)

    x = np.arange(len(cases), dtype=float)
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(10.0, len(cases) * 0.8), 6.5))

    for index, model in enumerate(models):
        rows = (
            metric_data[metric_data["model"] == model]
            .drop_duplicates("case", keep="last")
            .set_index("case")
            .reindex(cases)
        )
        means = rows["metric_mean"].to_numpy(dtype=float)
        errors = rows["metric_sd"].fillna(0).to_numpy(dtype=float)
        positions = x + (index - (len(models) - 1) / 2) * width
        ax.bar(
            positions,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_title(f"RQ3 model comparison — {metric}")
    ax.set_xlabel("Event log | Task")
    ax.set_ylabel(f"Mean {metric} (error bars: SD)")
    ax.set_xticks(x, cases, rotation=50, ha="right")
    ax.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        ncols=len(models),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))

    output_path = output_dir / f"rq3_comparison_{safe_filename(metric)}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    files = {
        "XGBoost": args.xgboost,
        "GNN": args.gnn,
        "Random Forest": args.random_forest,
        "LSTM": args.lstm,
        "LSTM Torch": args.lstm_torch,
        
    }
    results = load_results(files)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_metric(results, metric, args.output_dir, args.dpi)
        for metric in sorted(results["metric_name"].unique())
    ]
    print(f"Generated {len(outputs)} chart(s):")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
