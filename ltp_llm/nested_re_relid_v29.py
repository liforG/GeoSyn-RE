#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate V29 expanded candidates while retaining recursive children."""

from __future__ import print_function

try:
    from .ltp_relation_evidence_v29 import extract_relation_evidence, load_type_schema
    from .nested_re_relid_v22 import evidence_to_candidate
    from .nested_re_relid_v23 import _add_with_children
except (ImportError, ValueError):
    from ltp_relation_evidence_v29 import extract_relation_evidence, load_type_schema
    from nested_re_relid_v22 import evidence_to_candidate
    from nested_re_relid_v23 import _add_with_children


def generate_rows(enriched_rows, type_schema, enable_expansion=True):
    candidate_rows = []
    audit_rows = []
    for row in enriched_rows:
        audit = extract_relation_evidence(
            row, type_schema, enable_expansion=enable_expansion
        )
        triples = []
        seen = set()
        for item in audit["accepted"]:
            candidate = evidence_to_candidate(item)
            candidate["evidence"]["node"] = item.get("node")
            candidate["evidence"]["version"] = audit.get("version")
            _add_with_children(candidate, triples, seen)
        candidate_rows.append({
            "sid": audit["sid"],
            "sentence": audit["sentence"],
            "entities": [entity["text"] for entity in audit["entities"]],
            "triples": triples,
        })
        audit_rows.append(audit)
    return candidate_rows, audit_rows
