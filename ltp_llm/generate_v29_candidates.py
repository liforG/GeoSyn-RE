#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate formal V29 GNRED candidate graphs from LTP-enriched records."""

from __future__ import print_function

import argparse
import json
import os

try:
    from .ltp_relation_evidence_v29 import load_type_schema
    from .nested_re_relid_v22 import read_jsonl, write_jsonl
    from .nested_re_relid_v29 import generate_rows
except (ImportError, ValueError):
    from ltp_relation_evidence_v29 import load_type_schema
    from nested_re_relid_v22 import read_jsonl, write_jsonl
    from nested_re_relid_v29 import generate_rows


def relation_depth(node):
    if not isinstance(node, dict):
        return 0
    return 1 + max(relation_depth(node.get("subj")), relation_depth(node.get("obj")))


def main():
    parser = argparse.ArgumentParser(
        description="Generate formal V29 GNRED candidate graphs"
    )
    parser.add_argument("--input", required=True, help="LTP-enriched JSONL")
    parser.add_argument("--schema", required=True, help="GNRED operand schema JSON")
    parser.add_argument("--output", required=True, help="Candidate JSONL")
    parser.add_argument("--audit-output", help="Optional accepted/rejected evidence JSONL")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    schema = load_type_schema(args.schema)
    candidate_rows, audit_rows = generate_rows(rows, schema, enable_expansion=True)
    write_jsonl(args.output, candidate_rows)
    if args.audit_output:
        write_jsonl(args.audit_output, audit_rows)

    total = 0
    by_layer = {}
    by_relation = {}
    for row in candidate_rows:
        for candidate in row.get("triples", []):
            total += 1
            layer = str(relation_depth(candidate))
            relation = str(candidate.get("rid"))
            by_layer[layer] = by_layer.get(layer, 0) + 1
            by_relation[relation] = by_relation.get(relation, 0) + 1

    print("[V29] rows={} candidates={}".format(len(candidate_rows), total))
    print("[V29] by_layer={}".format(json.dumps(by_layer, sort_keys=True)))
    print("[V29] by_relation={}".format(
        json.dumps(by_relation, ensure_ascii=False, sort_keys=True)
    ))
    print("output={}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
