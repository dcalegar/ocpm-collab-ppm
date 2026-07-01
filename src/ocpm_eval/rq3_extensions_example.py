"""
Simple worked example for X-PaL and X-MSt — the two object-enabled extensions
with no single-case counterpart (catalog.py, single_case_counterpart=False).

RQ3.tex keeps these out of the representative subset because the available logs
only lightly exercise asynchrony (complete send/receive pairs, short latencies),
which would make X-MSt nearly degenerate; the text explicitly leaves the door
open to reporting them "if at all, as exploratory". This script is that
exploratory demonstration: it reuses the exact same feature/label/CV machinery
as rq3_pipeline.run_one_log (nothing X-PaL/X-MSt-specific to maintain), just
scoped to these two tasks and written to its own CSV so it is never mistaken
for part of the core 24 (task, log) coverage claim.

Usage: python -m ocpm_eval.rq3_extensions_example
"""
from dataclasses import replace
from typing import Optional
import pandas as pd

from .config import ExperimentConfig
from .rq3_pipeline import run_rq3


def run_extensions_example(cfg: Optional[ExperimentConfig] = None) -> pd.DataFrame:
    cfg = cfg or ExperimentConfig()
    ext_cfg = replace(cfg, rq3_tasks=["X-PaL", "X-MSt"])
    df = run_rq3(ext_cfg, out_name="rq3_extensions_example.csv")
    print("\n[note] X-PaL/X-MSt are exploratory here (see RQ3.tex): the bundled "
          "logs barely exercise asynchrony, so metrics may be near-degenerate "
          "(e.g. X-MSt close to constant). This demonstrates end-to-end "
          "feature-extractability, not a predictive claim.")
    return df


if __name__ == "__main__":
    run_extensions_example()
