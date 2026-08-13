"""
Leakage tests for the RQ3 object-centric feature families (A7, point 3),
complementing the remaining-time alignment oracle in
``features.ocpa.extract_feature_table`` (which only catches an
event-id/partitioning mismatch, not a leaking feature).

Covers every feature family in ``features.ocpa.build_feature_set`` --
elapsed time, previous_type_count(OrchestrationCase),
previous_type_count(Message), and preceding_activities(a) -- against two
leakage directions:

  * TEMPORAL: a feature value at cut point i must not depend on events
    after i within the same execution. Checked by comparing feature rows
    for the same case computed from a full log vs. from the same log
    with its last event removed: rows for the surviving cut points must
    be byte-for-byte identical.

  * CROSS-CASE: a feature value for an event in case C1 must not depend
    on other CollaborationCases in the log, even when they share a
    Participant identifier (rule M2 links Participant projections
    log-wide). Checked by comparing C1's feature rows computed alone vs.
    computed in a log where C2 shares C1's "Hospital" participant.

This generalizes, as a permanent regression test, the ad-hoc verification
done for A7 point 1 (which covered only previous_type_count and was
built/discarded in-session) to all feature families and to both leakage
directions.

Fixtures are pre-built SQLite files under tests/fixtures/ (see
src/mapping/support/build_leakage_test_logs.py) because building an OCEL 2.0
SQLite requires pm4py>=2.7 (the .venv-mapping environment), while reading
it back through OCPA for feature extraction requires the separate ocpa
environment (.venv) -- the same two-venv split the converter/evaluation
pipeline uses end-to-end (see src/mapping/README.md).

Run (ocpa venv):
    arch -x86_64 .venv/bin/python3.10 tests/test_features_leakage.py
(also pytest-compatible, once pytest is installed in that venv)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.io_ocel import load_ocpa_ocel, read_ocel2_labels     # noqa: E402
from features.ocpa import extract_feature_table                    # noqa: E402
from tasks.schema import Schema                                # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _extract(fixture_name):
    path = os.path.join(FIXTURES, fixture_name)
    schema = Schema()
    ocpa_ocel = load_ocpa_ocel(schema, path)
    ocel_log = read_ocel2_labels(path, schema, ocpa_ocel=ocpa_ocel)
    feats = extract_feature_table(fixture_name, schema, path, ocel_log, ocpa_ocel=ocpa_ocel)
    return feats["table"], feats["feature_cols"]


def test_no_temporal_leakage():
    """Dropping the last event of C1 must not change the feature rows of
    the events that remain: a feature at cut point i is a function of the
    prefix up to i, never of what comes after."""
    full, cols_full = _extract("leak_temporal_full.sqlite")
    trunc, cols_trunc = _extract("leak_temporal_truncated.sqlite")

    assert len(full) == 6 and len(trunc) == 5
    common_cols = [c for c in cols_full if c in cols_trunc]
    assert len(common_cols) >= 4  # elapsed_time + 2 previous_type_count + >=1 preceding_activities
    surviving_ids = set(trunc["event_id"])

    a = (full[full["event_id"].isin(surviving_ids)]
         .sort_values("event_id")[["event_id"] + common_cols].reset_index(drop=True))
    b = trunc.sort_values("event_id")[["event_id"] + common_cols].reset_index(drop=True)
    assert a.equals(b), (
        "feature rows for surviving cut points changed after removing a "
        "later event -- a feature is leaking information from the future")


def test_feature_table_row_order_is_deterministic():
    """The table must come back sorted by (case_id, event_id), not in OCPA's
    traversal order. RandomForest draws bootstrap rows by position at a fixed
    random_state, so a reordered table shifts the reported metrics even when
    every feature value is identical -- and OCPA's order depends on the
    object-type names of the log, which rule M2 makes log-dependent."""
    table, _ = _extract("leak_crosscase_combined.sqlite")
    keys = list(zip(table["case_id"], table["event_id"]))
    assert keys == sorted(keys)
    # Within a case the event identifier embeds prec_L (M5), so this is the
    # per-case source order that P1.2 reconstructs.
    c1 = table[table["case_id"] == "C1"]["event_id"].tolist()
    assert c1 == sorted(c1) and len(c1) > 1


def test_no_crosscase_leakage_via_shared_participant():
    """C1's feature rows must be unchanged whether it is the only case in
    the log or shares its 'Hospital' participant identifier with C2 (rule
    M2 links Pa objects log-wide, but per-execution features must not
    cross the CollaborationCase boundary)."""
    single, cols_single = _extract("leak_crosscase_single.sqlite")
    combined, cols_comb = _extract("leak_crosscase_combined.sqlite")

    assert set(combined["case_id"]) == {"C1", "C2"}
    common = [c for c in cols_single if c in cols_comb]
    assert len(common) >= 4

    a = single.sort_values("event_id")[["event_id"] + common].reset_index(drop=True)
    b = (combined[combined["case_id"] == "C1"]
         .sort_values("event_id")[["event_id"] + common].reset_index(drop=True))
    assert a.equals(b), (
        "C1's feature rows changed when C2 (sharing C1's Participant) was "
        "added to the log -- a feature is leaking across CollaborationCases")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n[ok] {len(fns)} feature-leakage tests passed.")


if __name__ == "__main__":
    _run_all()
