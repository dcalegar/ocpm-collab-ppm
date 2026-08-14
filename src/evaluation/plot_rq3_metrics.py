"""Unified RQ3 report: accuracy metrics (rq3_results_*.csv) and, when present,
per-stage time/memory profiling (rq3_profile_*.csv, written when RQ3 is run
with cfg.profile=True / --profile -- see evaluation.profiling). Consistent
colors across every chart so a model is immediately recognizable regardless
of which chart it appears in.

Per-predictor CSV paths default to the naming convention `run_evaluation.py`
writes (`rq3_{results,profile}_{predictor}_{log_group}[_full].csv`, see its
module docstring), derived from --log-group/--scope/--results-dir; pass e.g.
--xgboost/--xgboost-profile explicitly to override a single model. A model
whose *default* path doesn't exist is skipped (not every predictor has been
run against every log group/scope); an explicitly-passed path that doesn't
exist still raises.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_KEYS = {
    "XGBoost": "xgboost",
    "GNN": "gnn",
    "Random Forest": "random_forest",
    "LSTM": "lstm",
    "LSTM Torch": "lstm_torch",
    "Transformer": "transformer",
}

# Fixed across every figure (accuracy AND profiling) so each model is
# immediately recognizable regardless of which chart it appears in.
MODEL_COLORS = {
    "XGBoost": "#E76F51",
    "GNN": "#2A9D8F",
    "Random Forest": "#457B9D",
    "LSTM": "#9B5DE5",
    "LSTM Torch": "#E9C46A",
    "Transformer": "#6D9F71",
}

REQUIRED_RESULTS_COLUMNS = {"log", "task", "metric_name", "metric_mean", "metric_sd"}
REQUIRED_PROFILE_COLUMNS = {"log", "stage", "seconds", "rss_peak_mb"}
PROFILE_STAGES = ["ocel_load", "feature_extraction", "labeling", "fit", "predict"]


def _rq3_filename(kind: str, predictor_key: str, log_group: str, scope: str) -> str:
    prefix = "rq3_results" if kind == "results" else "rq3_profile"
    suffix = "_full" if scope == "full" else ""
    return f"{prefix}_{predictor_key}_{log_group}{suffix}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the unified RQ3 report: one chart per accuracy "
                    "metric, plus (if rq3_profile_*.csv files are present) one "
                    "chart per pipeline stage for time and peak memory."
    )
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="directory holding rq3_results_*.csv / rq3_profile_*.csv "
                            "(default: data/results/{log-group}_{scope}, matching "
                            "run_evaluation.py's per-stage output directory)")
    parser.add_argument("--log-group", default="predictcollab",
                        choices=["predictcollab", "bpi2013"])
    parser.add_argument("--scope", default="full", choices=["partial", "full"],
                        help="RQ3 task catalog scope the CSVs were generated "
                            "with (default: full, the 14-task supplementary run)")
    for flag, model in [("--xgboost", "XGBoost"), ("--gnn", "GNN"),
                        ("--random-forest", "Random Forest"), ("--lstm", "LSTM"),
                        ("--lstm-torch", "LSTM Torch"),
                        ("--transformer", "Transformer")]:
        parser.add_argument(flag, type=Path, default=None,
                            help=f"override the results CSV for {model} "
                                "(default: derived from --results-dir/--log-group/"
                                "--scope; skipped if that file doesn't exist)")
        parser.add_argument(f"{flag}-profile", type=Path, default=None,
                            help=f"override the profile CSV for {model} "
                                "(default: derived the same way; skipped if absent)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="directory where PNG charts are saved (default: "
                            "a 'plots' subdirectory of --results-dir, i.e. "
                            "data/results/{log-group}_{scope}/plots)")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    stage = f"{args.log_group}_{args.scope}"
    if args.results_dir is None:
        args.results_dir = Path("data/results") / stage
    if args.output_dir is None:
        args.output_dir = args.results_dir / "plots"
    return args


def _resolve_paths(args: argparse.Namespace, kind: str) -> Dict[str, Tuple[Path, bool]]:
    """{model: (path, explicit)}. explicit=True means the user passed the flag,
    so a missing file should raise rather than be silently skipped."""
    explicit = {
        "XGBoost": args.xgboost if kind == "results" else args.xgboost_profile,
        "GNN": args.gnn if kind == "results" else args.gnn_profile,
        "Random Forest": args.random_forest if kind == "results" else args.random_forest_profile,
        "LSTM": args.lstm if kind == "results" else args.lstm_profile,
        "LSTM Torch": args.lstm_torch if kind == "results" else args.lstm_torch_profile,
        "Transformer": args.transformer if kind == "results" else args.transformer_profile,
    }
    out = {}
    for model, key in MODEL_KEYS.items():
        if explicit[model] is not None:
            out[model] = (explicit[model], True)
        else:
            fname = _rq3_filename(kind, key, args.log_group, args.scope)
            out[model] = (args.results_dir / fname, False)
    return out


def _load_tagged(paths: Dict[str, Tuple[Path, bool]],
                 required_columns: set) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for model, (path, explicit) in paths.items():
        if not path.is_file():
            if explicit:
                raise FileNotFoundError(f"CSV not found for {model}: {path}")
            print(f"[skip] {model}: {path} not found")
            continue
        frame = pd.read_csv(path)
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        frame = frame.copy()
        frame["model"] = model
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=sorted(required_columns) + ["model"])
    return pd.concat(frames, ignore_index=True)


def load_results(paths: Dict[str, Tuple[Path, bool]]) -> pd.DataFrame:
    results = _load_tagged(paths, REQUIRED_RESULTS_COLUMNS)
    if results.empty:
        raise ValueError("No rq3_results_*.csv could be loaded -- nothing to plot. "
                         "Check --results-dir/--log-group/--scope, or override a "
                         "single model's path explicitly.")
    results["metric_mean"] = pd.to_numeric(results["metric_mean"], errors="coerce")
    results["metric_sd"] = pd.to_numeric(results["metric_sd"], errors="coerce")
    results = results.dropna(subset=["metric_name", "metric_mean"])
    if results.empty:
        raise ValueError("The CSV files contain no valid metric values.")
    return results


def load_profile(paths: Dict[str, Tuple[Path, bool]]) -> pd.DataFrame:
    """Unlike load_results, an empty return is a normal outcome (profiling is
    opt-in), not an error -- callers should skip the profile section instead
    of failing the whole report."""
    profile = _load_tagged(paths, REQUIRED_PROFILE_COLUMNS)
    if profile.empty:
        return profile
    for col in ("seconds", "rss_peak_mb"):
        profile[col] = pd.to_numeric(profile[col], errors="coerce")
    return profile.dropna(subset=["stage", "seconds"])


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _ordered_models(data: pd.DataFrame) -> List[str]:
    present = set(data["model"].unique())
    return [m for m in MODEL_KEYS if m in present]


def _grouped_bar(data: pd.DataFrame, models: List[str], case_col: str,
                 mean_col: str, sd_col: Optional[str], title: str, xlabel: str,
                 ylabel: str, output_path: Path, dpi: int) -> Path:
    cases = list(dict.fromkeys(data[case_col]))
    x = np.arange(len(cases), dtype=float)
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(10.0, len(cases) * 0.8), 6.5))

    for index, model in enumerate(models):
        rows = (
            data[data["model"] == model]
            .drop_duplicates(case_col, keep="last")
            .set_index(case_col)
            .reindex(cases)
        )
        means = rows[mean_col].to_numpy(dtype=float)
        errors = rows[sd_col].fillna(0).to_numpy(dtype=float) if sd_col else None
        positions = x + (index - (len(models) - 1) / 2) * width
        ax.bar(
            positions, means, width, yerr=errors, capsize=3, label=model,
            color=MODEL_COLORS[model], edgecolor="white", linewidth=0.6,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, cases, rotation=50, ha="right")
    ax.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Model", ncols=len(models),
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric(data: pd.DataFrame, metric: str, models: List[str],
                output_dir: Path, dpi: int) -> Path:
    metric_data = data[data["metric_name"] == metric].copy()
    metric_data["case"] = (
        metric_data["log"].astype(str) + " | " + metric_data["task"].astype(str)
    )
    return _grouped_bar(
        metric_data, models, "case", "metric_mean", "metric_sd",
        title=f"RQ3 model comparison — {metric}",
        xlabel="Event log | Task", ylabel=f"Mean {metric} (error bars: SD)",
        output_path=output_dir / f"rq3_comparison_{safe_filename(metric)}.png", dpi=dpi,
    )


def aggregate_stage(profile: pd.DataFrame, stage: str) -> pd.DataFrame:
    """One row per (model, log, task): mean/sd of seconds and rss_peak_mb
    across whatever repetitions exist for that stage -- folds for "fit"/
    "predict" (recorded per fold by each predictors/*.py module via a bound
    StageTimer), a single measurement for the per-log/per-task stages
    (ocel_load, feature_extraction, labeling)."""
    sub = profile[profile["stage"] == stage]
    if sub.empty:
        return sub
    grouped = sub.groupby(["model", "log", "task"], dropna=False)
    agg = grouped.agg(
        seconds_mean=("seconds", "mean"), seconds_sd=("seconds", "std"),
        rss_peak_mean=("rss_peak_mb", "mean"), rss_peak_sd=("rss_peak_mb", "std"),
    ).reset_index()
    agg[["seconds_sd", "rss_peak_sd"]] = agg[["seconds_sd", "rss_peak_sd"]].fillna(0.0)
    agg["case"] = np.where(
        agg["task"].isna(), agg["log"].astype(str),
        agg["log"].astype(str) + " | " + agg["task"].astype(str),
    )
    return agg


def plot_stage(profile: pd.DataFrame, stage: str, models: List[str],
              output_dir: Path, dpi: int) -> List[Path]:
    agg = aggregate_stage(profile, stage)
    if agg.empty:
        return []
    stage_models = [m for m in models if m in set(agg["model"].unique())]
    if not stage_models:
        return []
    outputs = []
    outputs.append(_grouped_bar(
        agg, stage_models, "case", "seconds_mean", "seconds_sd",
        title=f"RQ3 profiling — {stage} (time)",
        xlabel="Event log | Task", ylabel="Mean seconds (error bars: SD)",
        output_path=output_dir / f"rq3_profile_{safe_filename(stage)}_seconds.png", dpi=dpi,
    ))
    outputs.append(_grouped_bar(
        agg, stage_models, "case", "rss_peak_mean", "rss_peak_sd",
        title=f"RQ3 profiling — {stage} (peak memory)",
        xlabel="Event log | Task", ylabel="Mean peak RSS, MB (error bars: SD)",
        output_path=output_dir / f"rq3_profile_{safe_filename(stage)}_memory.png", dpi=dpi,
    ))
    return outputs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(_resolve_paths(args, "results"))
    models = _ordered_models(results)
    outputs = [
        plot_metric(results, metric, models, args.output_dir, args.dpi)
        for metric in sorted(results["metric_name"].unique())
    ]
    print(f"Generated {len(outputs)} accuracy chart(s):")
    for output in outputs:
        print(f"- {output}")

    profile = load_profile(_resolve_paths(args, "profile"))
    if profile.empty:
        print("\nNo rq3_profile_*.csv found -- skipping time/memory charts "
             "(run RQ3 with --profile to generate them).")
        return
    profile_models = _ordered_models(profile)
    profile_outputs = [
        p for stage in PROFILE_STAGES
        for p in plot_stage(profile, stage, profile_models, args.output_dir, args.dpi)
    ]
    print(f"\nGenerated {len(profile_outputs)} profiling chart(s):")
    for output in profile_outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
