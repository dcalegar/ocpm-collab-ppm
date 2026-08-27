"""
Consistency test for the *target anchor* of the fourteen reformulated tasks.

The anchor is declared in three places that no code path keeps in sync:

  1. ``tasks.catalog.TASKS[...].anchor`` -- reporting metadata only (no label
     function reads it), which is exactly why it can drift unnoticed;
  2. the ``anchor`` column of every ``rq3_results_*.csv`` under ``data/results/``,
     written from (1) by ``evaluation.rq3_pipeline``;
  3. the ``Anchor`` column of the RQ3 result tables in the paper, which are
     written by hand.

They had drifted apart (NE-NMPa, NE-NMPr and OB-P disagreed across all three).
GOLDEN below is the single source of truth; it mirrors the anchor rule stated in
the paper's task section -- the anchor is what the label *denotes*, not the scope
its quantification is restricted to -- so NE-NMPa and NE-NMPr, which return a
message kind, are anchored on Message even though NE-NMPa restricts its search to
one participant's orchestration case.

Run:  python tests/test_task_anchors.py    (also pytest-compatible)
"""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tasks.catalog import TASKS  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

GOLDEN = {
    "NE-NEPr": "CollaborationCase",
    "NE-NPaA": "Participant",
    "NE-NEPa": "Participant",
    "NE-NPaM": "Participant",
    "NE-NMPa": "Message",
    "NE-NMPr": "Message",
    "NV-PrT":  "CollaborationCase",
    "NV-PaT":  "OrchestrationCase",
    "NV-TNE":  "CollaborationCase",
    "NV-TNM":  "Message",
    "NV-NMPr": "Message",
    "NV-NMPa": "Message",
    "OB-P":    "Participant",
    "OB-M":    "Message",
}

# Result tables actually \input by the compiled paper (the orphaned ones are not
# checked; see the repository-cleanup item of the revision plan).
PAPER_TABLES = (
    "tableRQ3AllResultsFullPart1.tex",
    "tableRQ3AllResultsFullPart2.tex",
    "tableRQ3AllResultsSubsetv2.tex",
)

_TASK_RE = re.compile(r"\{\}l@\{\}\}(N[EV]-[A-Za-z]+|OB-[A-Za-z]+)\\\\")
_ANCHOR_RE = re.compile(r"\\texttt\{(CollaborationCase|OrchestrationCase|Participant|Message)\}")


def _paper_dir():
    """The paper sources live in a sibling repository, not in this one."""
    return os.environ.get("PAPER_DIR", os.path.join(ROOT, "..", "InfSys-OCPMPredictCollab"))


def test_catalog_matches_golden():
    assert set(TASKS) == set(GOLDEN), (sorted(TASKS), sorted(GOLDEN))
    for key, task in sorted(TASKS.items()):
        assert task.anchor == GOLDEN[key], (key, task.anchor, GOLDEN[key])


def test_result_csvs_match_golden():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "results", "**",
                                          "rq3_results_*.csv"), recursive=True))
    assert files, "no rq3_results_*.csv found under data/results/"
    rows = 0
    for path in files:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows += 1
                assert row["anchor"] == GOLDEN[row["task"]], (
                    os.path.relpath(path, ROOT), row["task"],
                    row["anchor"], GOLDEN[row["task"]])
    assert rows, "result CSVs are empty"


def test_paper_result_tables_match_golden():
    """The Anchor column of the paper's RQ3 tables. Skipped when the paper
    repository is not checked out next to this one."""
    paper = _paper_dir()
    if not os.path.isdir(paper):
        print("  skip  paper sources not found at", paper)
        return
    checked = 0
    for name in PAPER_TABLES:
        path = os.path.join(paper, name)
        if not os.path.isfile(path):
            continue
        task = None
        for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
            found = _TASK_RE.search(line)
            if found:
                task = found.group(1)
            if task is None:
                continue
            for anchor in _ANCHOR_RE.findall(line):
                checked += 1
                assert anchor == GOLDEN[task], (name, lineno, task, anchor,
                                                GOLDEN[task])
    assert checked >= 3 * 5, f"parsed only {checked} anchor cells; regex out of date?"


def test_paper_representative_subset_table_matches_golden():
    """The task/anchor listing of the representative subset, whose one-row-per-task
    layout the multirow parser above does not cover."""
    path = os.path.join(_paper_dir(), "tableRepresentativeSubset.tex")
    if not os.path.isfile(path):
        print("  skip  paper sources not found at", path)
        return
    checked = 0
    for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
        found = re.match(r"\s*(N[EV]-[A-Za-z]+|OB-[A-Za-z]+)\s*&", line)
        if not found:
            continue
        anchors = _ANCHOR_RE.findall(line)
        assert len(anchors) == 1, (lineno, line)
        checked += 1
        assert anchors[0] == GOLDEN[found.group(1)], (
            lineno, found.group(1), anchors[0], GOLDEN[found.group(1)])
    assert checked == 6, f"expected 6 subset rows, parsed {checked}"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} task-anchor consistency tests passed.")


if __name__ == "__main__":
    _run_all()
