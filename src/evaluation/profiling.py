"""Opt-in, per-stage wall-clock + RSS memory profiling for the RQ3 pipeline.

Collected into a tidy list of rows and written to its own artifact
(rq3_profile_{predictor}_{log_group}[_full].csv), never merged into
rq3_results_*.csv -- see the "V4 profile: no computation times are reported"
note in rq3_pipeline.py's module docstring, which this preserves by keeping
the two files separate rather than overriding it.

Uses psutil (whole-process RSS), not tracemalloc: the profiled stages are
dominated by native-library memory (OCPA, scikit-learn, XGBoost, PyTorch,
DGL) that tracemalloc's Python-heap-only tracking would not see.
"""
import threading
import time
from contextlib import contextmanager
from typing import List, Optional

import pandas as pd
import psutil

_PROCESS = psutil.Process()
_POLL_SECONDS = 0.05


class ProfileCollector:
    """Accumulates one row per profiled stage. `log`/`predictor` are always
    set; `task`/`fold` are None for per-log stages (ocel_load,
    feature_extraction), set for the per-task stage (labeling) and per-fold
    stages (fit, predict -- recorded by each predictors/*.py module via a
    bound StageTimer) respectively."""

    def __init__(self) -> None:
        self.rows: List[dict] = []

    @contextmanager
    def stage(self, stage: str, log: str, predictor: str,
             task: Optional[str] = None, fold: Optional[int] = None):
        # Background poller so a peak reached mid-stage isn't missed just
        # because it was freed again before the stage returned -- a plain
        # before/after delta can silently show ~0 even for a real spike.
        samples = [_PROCESS.memory_info().rss]
        stop = threading.Event()

        def poll():
            while not stop.wait(_POLL_SECONDS):
                samples.append(_PROCESS.memory_info().rss)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            stop.set()
            poller.join(timeout=_POLL_SECONDS * 4)
            samples.append(_PROCESS.memory_info().rss)
            self.rows.append({
                "log": log, "predictor": predictor, "task": task, "fold": fold,
                "stage": stage, "seconds": round(elapsed, 4),
                "rss_start_mb": round(samples[0] / 1e6, 2),
                "rss_end_mb": round(samples[-1] / 1e6, 2),
                "rss_peak_mb": round(max(samples) / 1e6, 2),
            })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class NullProfileCollector:
    """No-op stand-in used when cfg.profile is False, so run_one_log can call
    profiler.stage(...) unconditionally instead of branching on cfg.profile
    at every call site -- zero timing/threading/memory-sampling overhead."""

    @contextmanager
    def stage(self, stage: str, log: str, predictor: str,
             task: Optional[str] = None, fold: Optional[int] = None):
        yield


class StageTimer:
    """Bound to one (log, predictor, task, fold) fit_and_score_fold call, so
    a predictors/*.py module can record its own "fit"/"predict" stages
    (`timer.stage("fit")`) without importing ProfileCollector or repeating
    that context. Passed as fit_and_score_fold's optional `timer` argument;
    works against either a real ProfileCollector or a NullProfileCollector
    (profiling off), so run_one_log always passes a StageTimer -- no
    separate null variant needed here. predictors/*.py's own default when no
    timer is passed at all (e.g. a direct unit-test call, see
    tests/test_predictors_registry.py) is predictors.common.NullStageTimer,
    which duck-types this same `.stage(name)` protocol without importing
    this module -- predictors stays decoupled from evaluation (see
    predictors/README.md)."""

    def __init__(self, collector: ProfileCollector, log: str, predictor: str,
                task: Optional[str] = None, fold: Optional[int] = None) -> None:
        self._collector, self._log, self._predictor = collector, log, predictor
        self._task, self._fold = task, fold

    @contextmanager
    def stage(self, name: str):
        with self._collector.stage(name, self._log, self._predictor,
                                   task=self._task, fold=self._fold):
            yield
