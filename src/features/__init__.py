"""
features — object-centric feature extraction for OCEL 2.0 logs, decoupled from any
particular predictor or evaluation pipeline. Reused by ``ocpm_eval``'s RQ3
pipeline, but importable on its own.

  io_ocel   OCEL 2.0 SQLite -> OCPA object (features) / neutral model (labels),
            sharing one read path (see ``load_ocpa_ocel`` / ``read_ocel2_labels``)
  ocpa      native OCPA past-relative event features + the remaining-time alignment
            oracle (``build_feature_set`` / ``extract_feature_table``)
"""
__all__ = ["io_ocel", "ocpa"]
