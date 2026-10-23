#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate V23 GNRED candidates from bottom-up LTP evidence nodes."""

from __future__ import print_function

import argparse
import json
import os

try:
    from .ltp_relation_evidence_v23 import (
        extract_relation_evidence,
        load_type_schema,
    )
    from .nested_re_relid_v22 import (
        evidence_to_candidate,
        read_jsonl,
        write_jsonl,
    )
except (ImportError, ValueError):
    from ltp_relation_evidence_v23 import (
        extract_relation_evidence,
        load_type_schema,
    )
    from nested_re_relid_v22 import (
        evidence_to_candidate,
        read_jsonl,
        write_jsonl,
    )


def _canonical(node):
    if not isinstance(node, dict):
        return ("entity", str(node))
    return (
        "relation",
        str(node.get("rid")),
        _canonical(node.get("subj")),
        _canonical(node.get("obj")),
    )


def _add_with_children(node, triples, seen):
    if isinstance(node.get("subj"), dict):
        _add_with_children(node["subj"], triples, seen)
    if isinstance(node.get("obj"), dict):
        _add_with_children(node["obj"], triples, seen)
    key = _canonical(node)
    if key not in seen:
        seen.add(key)
        triples.append(node)


def generate_rows(enriched_rows, type_schema, enable_composition=True):
    candidate_rows = []
    audit_rows = []
    for row in enriched_rows:
        audit = extract_relation_evidence(
            row, type_schema, enable_composition=enable_composition
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate conservative V23 LTP candidates."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--type-file", default="data/type3.json")
    parser.add_argument("--output-ids", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--disable-composition", action="store_true")
    args = parser.parse_args()

    schema = load_type_schema(args.type_file)
    candidate_rows, audit_rows = generate_rows(
        read_jsonl(args.input),
        schema,
        enable_composition=not args.disable_composition,
    )
    write_jsonl(args.output_ids, candidate_rows)
    write_jsonl(args.audit, audit_rows)

    accepted = sum(len(row["accepted"]) for row in audit_rows)
    rejected = sum(len(row["rejected"]) for row in audit_rows)
    patterns = {}
    for row in audit_rows:
        for item in row["accepted"]:
            pattern = item.get("pattern", "<missing>")
            patterns[pattern] = patterns.get(pattern, 0) + 1
    print("V23 rows={} accepted={} rejected={}".format(
        len(candidate_rows), accepted, rejected
    ))
    print("patterns:", json.dumps(patterns, ensure_ascii=False, sort_keys=True))
    print("candidates:", os.path.abspath(args.output_ids))
    print("audit:", os.path.abspath(args.audit))


if __name__ == "__main__":
    main()
