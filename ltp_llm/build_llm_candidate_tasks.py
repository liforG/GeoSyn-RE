#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build sentence-grouped LLM scoring tasks from V21 candidate triples."""

import argparse
import hashlib
import json
import os
import itertools


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
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def relation_id(node):
    for key in ("rid", "_rid", "relation"):
        if key in node:
            return str(node[key])
    # Text-only candidates use rel; ID candidates should normally use rid.
    if "rel" in node:
        return str(node["rel"])
    raise ValueError("relation node has no relation ID: {!r}".format(node))


def canonical_node(node):
    if not isinstance(node, dict):
        return ["ENTITY", str(node)]

    def canonical_operand(side):
        value = node.get(side)
        if isinstance(value, dict):
            return canonical_node(value)
        entity_id = (node.get("evidence") or {}).get(
            "{}_entity_id".format("left" if side == "subj" else "right")
        )
        if entity_id is not None:
            return ["ENTITY_ID", str(entity_id)]
        return ["ENTITY", str(value)]

    return [
        "RELATION",
        relation_id(node),
        canonical_operand("subj"),
        canonical_operand("obj"),
    ]


def canonical_key(node):
    return json.dumps(canonical_node(node), ensure_ascii=False, separators=(",", ":"))


def relation_layer(node):
    if not isinstance(node, dict):
        return 0
    return 1 + max(relation_layer(node.get("subj")), relation_layer(node.get("obj")))


def short_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_candidate_groups(rows):
    groups = []
    seen_node_ids = set()

    for row_id, row in enumerate(rows):
        sentence = str(row.get("sentence", ""))
        sid = str(row.get("sid", row_id))
        group_id = "g_{}".format(short_hash("{}\n{}".format(sid, sentence)))
        triples = row.get("triples", [])

        key_order = []
        key_to_triple_ids = {}
        for triple_id, triple in enumerate(triples):
            key = canonical_key(triple)
            if key not in key_to_triple_ids:
                key_order.append(key)
                key_to_triple_ids[key] = []
            key_to_triple_ids[key].append(triple_id)

        node_ids = {}
        for key in key_order:
            node_id = "n_{}".format(short_hash("{}\n{}".format(group_id, key)))
            if node_id in seen_node_ids:
                raise ValueError("node ID collision: {}".format(node_id))
            seen_node_ids.add(node_id)
            node_ids[key] = node_id

        candidates = []
        for key in key_order:
            source_triple_ids = key_to_triple_ids[key]
            triple_id = source_triple_ids[0]
            triple = triples[triple_id]
            direct_children = []
            for side in ("subj", "obj"):
                child = triple.get(side)
                if not isinstance(child, dict):
                    continue
                child_key = canonical_key(child)
                if child_key not in key_to_triple_ids:
                    raise ValueError(
                        "row {} sid {} candidate {} has an unlisted {} child: {}".format(
                            row_id, sid, triple_id, side, child_key
                        )
                    )
                direct_children.append(node_ids[child_key])

            candidates.append({
                "task_id": "{}:{}".format(row_id, triple_id),
                "node_id": node_ids[key],
                "source_triple_id": triple_id,
                "source_triple_ids": source_triple_ids,
                "source_task_ids": [
                    "{}:{}".format(row_id, source_id) for source_id in source_triple_ids
                ],
                "layer": relation_layer(triple),
                "direct_children": direct_children,
                "triple": triple,
            })

        groups.append({
            "group_id": group_id,
            "source_row_id": row_id,
            "sid": sid,
            "sentence": sentence,
            "entities": row.get("entities", []),
            "candidates": candidates,
        })

    return groups


def index_rows_by_sid(rows, label):
    indexed = {}
    for row in rows:
        sid = str(row.get("sid"))
        if sid in indexed:
            raise ValueError("duplicate {} SID: {}".format(label, sid))
        indexed[sid] = row
    return indexed


def load_type_schema(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return {
        str(relation_id): operand_rules
        for relation_id, operand_rules in data.get("relation", [])
    }


def entity_types_by_text(entities):
    result = {}
    for entity in entities:
        if not isinstance(entity, dict) or entity.get("text") is None:
            continue
        text = str(entity["text"])
        entity_type = str(entity.get("type", "<UNKNOWN>"))
        result.setdefault(text, set()).add(entity_type)
    return result


def node_output_types(node, types_by_text):
    if isinstance(node, dict):
        return {relation_id(node)}
    return set(types_by_text.get(str(node), {"<UNKNOWN>"}))


def collect_schema_checks(node, types_by_text, type_schema, output):
    if not isinstance(node, dict):
        return
    relation = relation_id(node)
    left = node.get("subj")
    right = node.get("obj")
    left_types = node_output_types(left, types_by_text)
    right_types = node_output_types(right, types_by_text)
    rules = type_schema.get(relation, [])
    allowed = any(
        left_type in rule[0] and right_type in rule[1]
        for left_type, right_type in itertools.product(
            left_types, right_types
        )
        for rule in rules
    )
    signature = (
        relation,
        tuple(sorted(left_types)),
        tuple(sorted(right_types)),
        bool(allowed),
    )
    output.add(signature)
    collect_schema_checks(left, types_by_text, type_schema, output)
    collect_schema_checks(right, types_by_text, type_schema, output)


def token_records(tokens, pos_values):
    return [
        {
            "token_id": index + 1,
            "text": token,
            "pos": pos_values[index] if index < len(pos_values) else None,
        }
        for index, token in enumerate(tokens)
    ]


def dependency_records(tokens, dependencies):
    result = []
    for item in dependencies:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("invalid dependency edge: {!r}".format(item))
        dependent_id = int(item[0])
        head_id = int(item[1])
        if dependent_id < 1 or dependent_id > len(tokens):
            raise ValueError(
                "dependency dependent index out of range: {!r}".format(item)
            )
        if head_id < 0 or head_id > len(tokens):
            raise ValueError(
                "dependency head index out of range: {!r}".format(item)
            )
        result.append({
            "dependent_id": dependent_id,
            "dependent": tokens[dependent_id - 1],
            "relation": str(item[2]),
            "head_id": head_id,
            "head": "ROOT" if head_id == 0 else tokens[head_id - 1],
        })
    return result


def _hypothesis_operand(value):
    if not isinstance(value, dict):
        return str(value)
    return {
        "relation": relation_id(value),
        "left": _hypothesis_operand(value.get("subj")),
        "right": _hypothesis_operand(value.get("obj")),
    }


def candidate_evidence_hypotheses(group):
    """Expose candidate-source evidence without score labels or Gold metadata."""
    result = []
    seen = set()
    for candidate in group.get("candidates", []):
        triple = candidate.get("triple", {})
        evidence = triple.get("evidence") or {}
        item = {
            "relation": relation_id(triple),
            "left": _hypothesis_operand(triple.get("subj")),
            "right": _hypothesis_operand(triple.get("obj")),
            "cue": triple.get("rel_text"),
            "dependency_path": evidence.get("dependency_path"),
        }
        item = {
            key: value for key, value in item.items()
            if value is not None and value != ""
        }
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def attach_ltp_and_schema(
    groups,
    enriched_rows,
    type_schema,
    candidate_evidence_as_hypotheses=False,
):
    """Attach the fixed v4 model evidence fields to sentence groups."""
    enriched_by_sid = index_rows_by_sid(enriched_rows, "enriched")
    group_sids = {str(group.get("sid")) for group in groups}
    missing = sorted(group_sids - set(enriched_by_sid))
    extra = sorted(set(enriched_by_sid) - group_sids)
    if missing or extra:
        raise ValueError(
            "candidate/enriched SID mismatch: missing={} extra={}".format(
                missing[:20], extra[:20]
            )
        )

    for group in groups:
        sid = str(group.get("sid"))
        enriched = enriched_by_sid[sid]
        if str(enriched.get("sentence", "")) != str(group.get("sentence", "")):
            raise ValueError(
                "SID {} candidate/enriched sentence mismatch".format(sid)
            )
        tokens = list(enriched.get("tokens", []))
        pos_values = list(enriched.get("pos", []))
        if len(tokens) != len(pos_values):
            raise ValueError(
                "SID {} token/POS length mismatch: {} vs {}".format(
                    sid, len(tokens), len(pos_values)
                )
            )

        entities = list(enriched.get("entities", []))
        types_by_text = entity_types_by_text(entities)
        schema_signatures = set()
        for candidate in group.get("candidates", []):
            collect_schema_checks(
                candidate["triple"],
                types_by_text,
                type_schema,
                schema_signatures,
            )

        group["entities"] = entities
        group["tokens"] = token_records(tokens, pos_values)
        group["dependency_edges"] = dependency_records(
            tokens, enriched.get("dep", [])
        )
        hypotheses = list(enriched.get("triggers", []))
        if candidate_evidence_as_hypotheses:
            hypotheses.extend(candidate_evidence_hypotheses(group))
        group["rule_hypotheses_may_be_wrong"] = hypotheses
        group["schema_checks"] = [
            {
                "relation": relation,
                "left_operand_types": list(left_types),
                "right_operand_types": list(right_types),
                "allowed_by_schema": allowed,
            }
            for relation, left_types, right_types, allowed
            in sorted(schema_signatures)
        ]
    return groups


def validate_candidate_groups(groups, source_rows=None):
    task_ids = set()
    node_ids = set()
    source_task_ids_seen = set()
    candidate_count = 0
    source_candidate_count = 0

    for group_index, group in enumerate(groups):
        local_nodes = set()
        candidates = group.get("candidates", [])
        for candidate in candidates:
            candidate_count += 1
            task_id = str(candidate.get("task_id", ""))
            node_id = str(candidate.get("node_id", ""))
            if not task_id or task_id in task_ids:
                raise ValueError("missing or duplicate task_id: {!r}".format(task_id))
            if not node_id or node_id in node_ids:
                raise ValueError("missing or duplicate node_id: {!r}".format(node_id))
            task_ids.add(task_id)
            node_ids.add(node_id)
            local_nodes.add(node_id)
            source_task_ids = candidate.get("source_task_ids", [])
            if not source_task_ids or task_id != str(source_task_ids[0]):
                raise ValueError("candidate {} has invalid source_task_ids".format(task_id))
            for source_task_id in source_task_ids:
                source_task_id = str(source_task_id)
                if source_task_id in source_task_ids_seen:
                    raise ValueError("duplicate source task alias: {}".format(source_task_id))
                source_task_ids_seen.add(source_task_id)
            source_candidate_count += len(source_task_ids)

        for candidate in candidates:
            for child_id in candidate.get("direct_children", []):
                if child_id not in local_nodes:
                    raise ValueError(
                        "group {} candidate {} references absent child {}".format(
                            group.get("group_id"), candidate.get("node_id"), child_id
                        )
                    )
                child = next(item for item in candidates if item["node_id"] == child_id)
                if int(child["layer"]) >= int(candidate["layer"]):
                    raise ValueError(
                        "child {} layer must be below parent {} layer".format(
                            child_id, candidate.get("node_id")
                        )
                    )

        if source_rows is not None:
            row_id = int(group.get("source_row_id", -1))
            if row_id != group_index or row_id >= len(source_rows):
                raise ValueError("group/source row mismatch at group {}".format(group_index))
            row_source_ids = []
            for candidate in candidates:
                source_triple_ids = [int(value) for value in candidate["source_triple_ids"]]
                row_source_ids.extend(source_triple_ids)
                for triple_id in source_triple_ids:
                    if canonical_key(candidate["triple"]) != canonical_key(
                            source_rows[row_id]["triples"][triple_id]):
                        raise ValueError(
                            "candidate/source structure mismatch: {}:{}".format(row_id, triple_id)
                        )
            expected_ids = list(range(len(source_rows[row_id].get("triples", []))))
            if sorted(row_source_ids) != expected_ids:
                raise ValueError("candidate aliases do not cover source row {}".format(row_id))

    return {
        "groups": len(groups),
        "candidates": candidate_count,
        "source_candidates": source_candidate_count,
        "deduplicated_aliases": source_candidate_count - candidate_count,
    }


def source_task_aliases(groups):
    """Map every source row:triple task to its semantically unique scoring task."""
    aliases = {}
    for group in groups:
        for candidate in group.get("candidates", []):
            scoring_task_id = str(candidate["task_id"])
            for source_task_id in candidate.get("source_task_ids", []):
                source_task_id = str(source_task_id)
                if source_task_id in aliases:
                    raise ValueError("duplicate source task alias: {}".format(source_task_id))
                aliases[source_task_id] = scoring_task_id
    return aliases


def iter_group_batches(groups, max_candidates):
    """Batch whole sentence groups without ever splitting a group."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    batch = []
    count = 0
    for group in groups:
        group_count = len(group.get("candidates", []))
        if group_count == 0:
            continue
        if batch and count + group_count > max_candidates:
            yield batch
            batch = []
            count = 0
        batch.append(group)
        count += group_count
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(
        description="Build sentence-grouped candidate graphs for web/API LLM scoring."
    )
    parser.add_argument("--input", required=True, help="V21 triples_ids JSONL path.")
    parser.add_argument(
        "--enriched",
        help="Matching LTP enriched JSONL; required with --type-file.",
    )
    parser.add_argument(
        "--type-file",
        help="Relation operand schema JSON; required with --enriched.",
    )
    parser.add_argument(
        "--candidate-evidence-as-hypotheses",
        action="store_true",
        help=(
            "Expose each candidate's cue and dependency path as "
            "auditable rule hypotheses. Intended for candidate generators "
            "whose enriched input has no trigger records."
        ),
    )
    parser.add_argument("--output", required=True, help="Output sentence-grouped task JSONL path.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    groups = build_candidate_groups(rows)
    if bool(args.enriched) != bool(args.type_file):
        parser.error("--enriched and --type-file must be provided together")
    if args.enriched:
        groups = attach_ltp_and_schema(
            groups,
            read_jsonl(args.enriched),
            load_type_schema(args.type_file),
            candidate_evidence_as_hypotheses=(
                args.candidate_evidence_as_hypotheses
            ),
        )
    summary = validate_candidate_groups(groups, source_rows=rows)
    write_jsonl(args.output, groups)

    layer_counts = {}
    edge_count = 0
    for group in groups:
        for candidate in group["candidates"]:
            layer = str(candidate["layer"])
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            edge_count += len(candidate["direct_children"])

    print("groups:", summary["groups"])
    print("candidates:", summary["candidates"])
    print("source_candidates:", summary["source_candidates"])
    print("deduplicated_aliases:", summary["deduplicated_aliases"])
    print("layers:", json.dumps(layer_counts, ensure_ascii=False, sort_keys=True))
    print("parent_child_edges:", edge_count)
    print("output:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
