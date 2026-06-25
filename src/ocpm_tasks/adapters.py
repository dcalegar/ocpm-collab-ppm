"""
Adapters that build the neutral object-centric model from a concrete OCEL, so that the
task library can be used **inside OCPA or pm4py** without coupling to the
experimentation. The collaborative roles (mapping.tex) are derived here from E2O
qualifiers (within/local/send/receive) and O2O qualifiers (executed_by/from/to);
``build_from_relations`` contains the (library-tested) derivation, and the
``from_pm4py`` / ``from_ocpa`` wrappers only extract relations and attributes from the
respective object.

VERIFY against your installed libraries: the pm4py OCEL column names (``C_*`` below)
and the OCPA OCEL accessors. They follow the standard column conventions.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd

from .model import Event, Execution, ObjectCentricLog
from .schema import Schema


# --- low-level neutral relation carriers (what the adapters fill) ------------
@dataclass
class _Ev:
    id: str
    activity: str
    time: datetime
    attrs: Dict[str, str] = field(default_factory=dict)
    e2o: List[Tuple[str, str]] = field(default_factory=list)   # (object_id, qualifier)


@dataclass
class _Ob:
    id: str
    otype: str
    attrs: Dict[str, str] = field(default_factory=dict)
    o2o: List[Tuple[str, str]] = field(default_factory=list)   # (object_id, qualifier)


def build_from_relations(events: List[_Ev], objects: Dict[str, _Ob],
                         schema: Optional[Schema] = None) -> ObjectCentricLog:
    sch = schema or Schema()

    def otype(oid):
        o = objects.get(oid) if oid else None
        return o.otype if o else None

    def oattr(oid, key):
        o = objects.get(oid) if oid else None
        return o.attrs.get(key) if o else None

    def o2o_target(src, qualifier, target_type):
        o = objects.get(src) if src else None
        if not o:
            return None
        for tgt, q in o.o2o:
            if q == qualifier and otype(tgt) == target_type:
                return tgt
        return None

    def participant_of(lc_oid, pa_direct):
        if pa_direct is not None:
            return str(oattr(pa_direct, sch.oa_name) or pa_direct)
        pa = o2o_target(lc_oid, sch.q_executed_by, sch.ot_participant)
        if pa is not None:
            return str(oattr(pa, sch.oa_name) or pa)
        v = oattr(lc_oid, sch.oa_participant)
        return str(v) if v is not None else ""

    def msg_party(msg_oid, qualifier):
        p = o2o_target(msg_oid, qualifier, sch.ot_participant)
        return str(oattr(p, sch.oa_name) or p) if p is not None else None

    by_ci: Dict[str, List[Event]] = {}
    ci_caseid: Dict[str, str] = {}

    for ev in events:
        ci_oid = lc_oid = msg_oid = pa_oid = None
        is_send = is_recv = False
        for oid, q in ev.e2o:
            t = otype(oid)
            if q == sch.q_within and t == sch.ot_ci:
                ci_oid = oid
            elif q == sch.q_local and t == sch.ot_lc:
                lc_oid = oid
            elif q == sch.q_send and t == sch.ot_message:
                is_send, msg_oid = True, oid
            elif q == sch.q_receive and t == sch.ot_message:
                is_recv, msg_oid = True, oid
            elif t == sch.ot_participant:
                pa_oid = oid
        if not (is_send or is_recv):
            elem = ev.attrs.get(sch.ea_elemtype)
            if elem == sch.elem_send:
                is_send = True
            elif elem == sch.elem_receive:
                is_recv = True
        if ci_oid is None:
            continue
        if ci_oid not in ci_caseid:
            ci_caseid[ci_oid] = str(oattr(ci_oid, sch.oa_caseid) or ci_oid)
        is_msg = is_send or is_recv
        by_ci.setdefault(ci_oid, []).append(Event(
            event_id=str(ev.id), activity=str(ev.activity), timestamp=ev.time,
            actor=participant_of(lc_oid, pa_oid),
            is_send=is_send, is_receive=is_recv,
            msg_id=str(msg_oid) if (is_msg and msg_oid is not None) else None,
            msg_type=(str(ev.activity) if is_msg else None),
            msg_from=msg_party(msg_oid, sch.q_from) if is_msg else None,
            msg_to=msg_party(msg_oid, sch.q_to) if is_msg else None))

    return ObjectCentricLog([Execution(ci_caseid[c], evs)
                             for c, evs in by_ci.items()])


# pm4py standard OCEL column names (VERIFY against your pm4py version).
C_EID, C_OID, C_OID2, C_OTYPE = "ocel:eid", "ocel:oid", "ocel:oid_2", "ocel:type"
C_QUAL, C_ACT, C_TS = "ocel:qualifier", "ocel:activity", "ocel:timestamp"


def from_pm4py(pm4py_ocel, schema: Optional[Schema] = None) -> ObjectCentricLog:
    """Optional adapter: build the neutral model from a pm4py OCEL object (read e.g.
    with pm4py.read_ocel2_json/sqlite). Requires pm4py >= 2.7 (NOT the pm4py==2.2.32
    pinned by OCPA, which cannot read OCEL 2.0); the default reader is
    from_ocel2_sqlite. Uses the relations/o2o/objects/events tables."""
    odf = pm4py_ocel.objects
    o_attr_cols = [c for c in odf.columns if c not in (C_OID, C_OTYPE)]
    o2o_map: Dict[str, List[Tuple[str, str]]] = {}
    o2o = getattr(pm4py_ocel, "o2o", None)
    if o2o is not None and len(o2o):
        for _, r in o2o.iterrows():
            o2o_map.setdefault(r[C_OID], []).append((r[C_OID2], r[C_QUAL]))
    objects: Dict[str, _Ob] = {}
    for _, r in odf.iterrows():
        attrs = {c: r[c] for c in o_attr_cols if pd.notna(r[c])}
        objects[r[C_OID]] = _Ob(r[C_OID], r[C_OTYPE], attrs, o2o_map.get(r[C_OID], []))

    e2o_map: Dict[str, List[Tuple[str, str]]] = {}
    for _, r in pm4py_ocel.relations.iterrows():
        e2o_map.setdefault(r[C_EID], []).append((r[C_OID], r[C_QUAL]))
    edf = pm4py_ocel.events
    e_attr_cols = [c for c in edf.columns if c not in (C_EID, C_ACT, C_TS)]
    events: List[_Ev] = []
    for _, r in edf.iterrows():
        attrs = {c: r[c] for c in e_attr_cols if pd.notna(r[c])}
        events.append(_Ev(r[C_EID], r[C_ACT],
                          pd.to_datetime(r[C_TS]).to_pydatetime(),
                          attrs, e2o_map.get(r[C_EID], [])))
    return build_from_relations(events, objects, schema)


def from_ocpa(ocpa_ocel, schema: Optional[Schema] = None) -> ObjectCentricLog:
    """Best-effort build from an OCPA-loaded OCEL object. The default reader for the
    label side is ``from_ocel2_sqlite`` (it reads the OCEL 2.0 SQLite directly and
    keeps E2O/O2O qualifiers). Implement this only if you prefer to reuse an
    already-imported OCPA OCEL. VERIFY the OCPA accessors for your version."""
    raise NotImplementedError(
        "from_ocpa: use from_ocel2_sqlite (default) or from_pm4py (pm4py >= 2.7).")


# --- OCEL 2.0 SQLite reader (stdlib sqlite3; version-independent) -------------
# Reads the OCEL 2.0 relational (SQLite) format directly, per the OCEL 2.0
# specification (tables: event, object, event_object[ocel_qualifier],
# object_object[ocel_qualifier], event_map_type/object_map_type, and the per-type
# tables event_<map>/object_<map> carrying ocel_time and attributes). This avoids a
# pm4py dependency for reading (OCPA pins an old pm4py that cannot read OCEL 2.0) and
# exposes E2O/O2O qualifiers needed to derive the collaborative roles.
def from_ocel2_sqlite(path: str, schema: "Schema | None" = None) -> ObjectCentricLog:
    import sqlite3
    from datetime import datetime

    def parse_time(v):
        if isinstance(v, datetime):
            return v
        s = str(v).strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return pd.to_datetime(v).to_pydatetime()

    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    def table_columns(tbl):
        return [r[1] for r in cur.execute(f'PRAGMA table_info("{tbl}")').fetchall()]

    et_map = {r["ocel_type"]: r["ocel_type_map"]
              for r in cur.execute("SELECT ocel_type, ocel_type_map FROM event_map_type")}
    ot_map = {r["ocel_type"]: r["ocel_type_map"]
              for r in cur.execute("SELECT ocel_type, ocel_type_map FROM object_map_type")}

    ev_type = {r["ocel_id"]: r["ocel_type"]
               for r in cur.execute("SELECT ocel_id, ocel_type FROM event")}
    ev_time, ev_attrs = {}, {}
    for suffix in set(et_map.values()):
        tbl = f"event_{suffix}"
        cols = table_columns(tbl)
        attr_cols = [c for c in cols if c not in ("ocel_id", "ocel_time")]
        for r in cur.execute(f'SELECT * FROM "{tbl}"'):
            d = dict(r)
            ev_time[d["ocel_id"]] = parse_time(d["ocel_time"])
            ev_attrs[d["ocel_id"]] = {c: d[c] for c in attr_cols if d[c] is not None}

    ob_type = {r["ocel_id"]: r["ocel_type"]
               for r in cur.execute("SELECT ocel_id, ocel_type FROM object")}
    ob_attrs: Dict[str, Dict[str, str]] = {}
    ob_attr_t: Dict[str, Dict[str, str]] = {}
    for suffix in set(ot_map.values()):
        tbl = f"object_{suffix}"
        cols = table_columns(tbl)
        attr_cols = [c for c in cols
                     if c not in ("ocel_id", "ocel_time", "ocel_changed_field")]
        for r in cur.execute(f'SELECT * FROM "{tbl}"'):
            d = dict(r)
            oid = d["ocel_id"]
            t = str(d.get("ocel_time") or "")
            cur_a = ob_attrs.setdefault(oid, {})
            cur_t = ob_attr_t.setdefault(oid, {})
            for c in attr_cols:
                if d[c] is None:
                    continue
                if c not in cur_t or t >= cur_t[c]:   # keep latest value
                    cur_a[c] = d[c]
                    cur_t[c] = t

    e2o: Dict[str, list] = {}
    for r in cur.execute("SELECT ocel_event_id, ocel_object_id, ocel_qualifier "
                         "FROM event_object"):
        e2o.setdefault(r["ocel_event_id"], []).append(
            (r["ocel_object_id"], r["ocel_qualifier"]))
    o2o: Dict[str, list] = {}
    for r in cur.execute("SELECT ocel_source_id, ocel_target_id, ocel_qualifier "
                         "FROM object_object"):
        o2o.setdefault(r["ocel_source_id"], []).append(
            (r["ocel_target_id"], r["ocel_qualifier"]))
    con.close()

    objects = {oid: _Ob(oid, ob_type[oid], ob_attrs.get(oid, {}), o2o.get(oid, []))
               for oid in ob_type}
    events = [_Ev(eid, ev_type[eid], ev_time.get(eid),
                  ev_attrs.get(eid, {}), e2o.get(eid, []))
              for eid in ev_type]
    return build_from_relations(events, objects, schema)
