#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate conservative V22 GNRED candidates from normalized LTP evidence."""

from __future__ import print_function

import argparse
import json
import os

try:
    from .ltp_relation_evidence_v22 import extract_relation_evidence, load_type_schema
except (ImportError, ValueError):
    # Direct script execution (``python new/nested_re_relid_v22.py``).
    from ltp_relation_evidence_v22 import extract_relation_evidence, load_type_schema


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("{}:{} invalid JSON: {}".format(path, line_no, exc))
    return rows


def write_jsonl(path, rows):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _operand_to_candidate(operand):
    if operand.get("kind") != "relation":
        return operand["text"]
    return {
        "subj": _operand_to_candidate(operand["left"]),
        "rid": operand["relation"],
        "rel_text": operand["lexeme"],
        "obj": _operand_to_candidate(operand["right"]),
        "evidence": {
            "pattern": operand["pattern"],
            "trigger_token_id": operand["trigger_token_id"],
            "left_entity_id": operand["left"].get("entity_id"),
            "right_entity_id": operand["right"].get("entity_id"),
            "left_type": _operand_type(operand["left"]),
            "right_type": _operand_type(operand["right"]),
            "dependency_path": operand["dependency_path"],
        },
    }


def _operand_type(operand):
    if operand.get("kind") == "relation":
        return operand["relation"]
    return operand["type"]


def evidence_to_candidate(evidence):
    return {
        "subj": _operand_to_candidate(evidence["left"]),
        "rid": evidence["relation"],
        "rel_text": evidence["lexeme"],
        "obj": _operand_to_candidate(evidence["right"]),
        "evidence": {
            "pattern": evidence["pattern"],
            "trigger_token_id": evidence["trigger_token_id"],
            "left_entity_id": evidence["left"].get("entity_id"),
            "right_entity_id": evidence["right"].get("entity_id"),
            "left_type": _operand_type(evidence["left"]),
            "right_type": _operand_type(evidence["right"]),
            "dependency_path": evidence["dependency_path"],
        },
    }


def generate_rows(enriched_rows, type_schema):
    candidate_rows = []
    audit_rows = []
    for row in enriched_rows:
        audit = extract_relation_evidence(row, type_schema)
        triples = []
        seen = set()

        def canonical(node):
            if not isinstance(node, dict):
                return ("entity", str(node))
            return ("relation", str(node.get("rid")), canonical(node.get("subj")), canonical(node.get("obj")))

        def add_with_children(node):
            if isinstance(node.get("subj"), dict):
                add_with_children(node["subj"])
            if isinstance(node.get("obj"), dict):
                add_with_children(node["obj"])
            key = canonical(node)
            if key not in seen:
                seen.add(key)
                triples.append(node)

        for item in audit["accepted"]:
            add_with_children(evidence_to_candidate(item))
        candidate_rows.append({
            "sid": audit["sid"],
            "sentence": audit["sentence"],
            "entities": [entity["text"] for entity in audit["entities"]],
            "triples": triples,
        })
        audit_rows.append(audit)
    return candidate_rows, audit_rows


def main():
    parser = argparse.ArgumentParser(description="Generate conservative V22 LTP candidates.")
    parser.add_argument("--input", required=True, help="run_batch_2 enriched JSONL")
    parser.add_argument("--type-file", default="data/type3.json", help="read-only GNRED schema")
    parser.add_argument("--output-ids", required=True, help="candidate JSONL output")
    parser.add_argument("--audit", required=True, help="accepted/rejected evidence JSONL")
    args = parser.parse_args()

    schema = load_type_schema(args.type_file)
    candidate_rows, audit_rows = generate_rows(read_jsonl(args.input), schema)
    write_jsonl(args.output_ids, candidate_rows)
    write_jsonl(args.audit, audit_rows)

    accepted = sum(len(row["accepted"]) for row in audit_rows)
    rejected = sum(len(row["rejected"]) for row in audit_rows)
    unresolved = sum(len(row["unresolved_entities"]) for row in audit_rows)
    print("V22 rows={} accepted={} rejected={} unresolved_entities={}".format(
        len(candidate_rows), accepted, rejected, unresolved
    ))
    print("candidates:", os.path.abspath(args.output_ids))
    print("audit:", os.path.abspath(args.audit))


if __name__ == "__main__":
    main()
