#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V28 bottom-up closure for the frozen V27 explicit L1 inventory.

V28 adds no L1 extraction family.  It reruns the V26 parent and hierarchical
scope builders over V27, so newly accepted direction/position relations can
participate in L2/L3 without changing any V27 decision.
"""

from __future__ import print_function

try:
    from . import ltp_relation_evidence_v22 as v22
    from . import ltp_relation_evidence_v23 as v23
    from . import ltp_relation_evidence_v24 as v24
    from . import ltp_relation_evidence_v25 as v25
    from . import ltp_relation_evidence_v26 as v26
    from . import ltp_relation_evidence_v27 as v27
except (ImportError, ValueError):
    import ltp_relation_evidence_v22 as v22
    import ltp_relation_evidence_v23 as v23
    import ltp_relation_evidence_v24 as v24
    import ltp_relation_evidence_v25 as v25
    import ltp_relation_evidence_v26 as v26
    import ltp_relation_evidence_v27 as v27


load_type_schema = v27.load_type_schema
schema_allows = v27.schema_allows

POSITIONAL_RELATIONS = {"A", "BW", "C"}
DIRECTION_RELATIONS = {"N", "S", "E", "W", "NE", "SE", "NW", "SW"}
CONTACT_MARKERS = {"接触", "相连"}
DISRUPTION_MARKERS = {"侵吞", "横切", "切割", "破坏", "覆盖"}
OCCURRENCE_MARKERS = {"出露", "发育", "见", "可见", "展现", "分布"}
SCOPE_MARKERS = {"中", "内", "内部", "之中", "范围内"}
IDENTITY_MARKERS = {"属", "所指", "指", "称", "称为"}


def _relation_operand(item):
    return v26._relation_operand(item)


def _accept(type_schema, relation, lexeme, trigger_id, left, right,
            pattern, path, accepted, rejected):
    evidence = v22._evidence_record(
        relation, lexeme, trigger_id, left, right, pattern, path
    )
    v22._accept_if_valid(type_schema, evidence, accepted, rejected)


def _path_between(start_id, target_id, heads, dependency_relations,
                  max_steps=10):
    return v24._dependency_path_between(
        int(start_id), int(target_id), heads, dependency_relations,
        v25.EXPLICIT_PATH_RELATIONS | {"FOB", "RAD", "DBL"},
        max_steps=max_steps,
    )


def _depth_one_relations(base, allowed):
    return [
        item for item in base["accepted"]
        if item.get("relation") in allowed
        and int((item.get("node") or {}).get("depth", 0)) == 1
        and item.get("left", {}).get("kind") == "entity"
        and item.get("right", {}).get("kind") == "entity"
    ]


def _extract_contact_closure(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []
    for item in _depth_one_relations(
        base, POSITIONAL_RELATIONS | DIRECTION_RELATIONS
    ):
        trigger_clause = clause_ids[int(item["trigger_token_id"]) - 1]
        child = _relation_operand(item)
        child_ids = v24._operand_entity_ids(child)
        clause_entity_ids = {
            entity.get("entity_id") for entity in base["entities"]
            if entity.get("head_token_id") is not None
            and clause_ids[int(entity["head_token_id"]) - 1] == trigger_clause
        }
        if not clause_entity_ids.issubset(child_ids):
            continue
        markers = [
            (token_id, token) for token_id, token in enumerate(tokens, 1)
            if token in CONTACT_MARKERS
            and clause_ids[token_id - 1] == trigger_clause
        ]
        if len(markers) != 1:
            continue
        marker_id, marker = markers[0]
        path = _path_between(
            item["trigger_token_id"], marker_id,
            heads, dependency_relations, max_steps=8,
        )
        if path is None:
            continue
        _accept(
            type_schema, "EC", marker, marker_id, item["left"], child,
            "explicit_contact_relation_closure",
            list(path) + list(item.get("dependency_path", [])),
            accepted, rejected,
        )
    return accepted, rejected


def _extract_disruption_closure(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []
    for item in _depth_one_relations(base, DIRECTION_RELATIONS):
        trigger_clause = clause_ids[int(item["trigger_token_id"]) - 1]
        markers = [
            (token_id, token) for token_id, token in enumerate(tokens, 1)
            if token in DISRUPTION_MARKERS
            and clause_ids[token_id - 1] == trigger_clause
        ]
        if len(markers) != 1:
            continue
        marker_id, marker = markers[0]
        path = _path_between(
            item["trigger_token_id"], marker_id,
            heads, dependency_relations, max_steps=8,
        )
        if path is None:
            continue
        child = _relation_operand(item)
        _accept(
            type_schema, "PO", marker, marker_id, item["left"], child,
            "explicit_direction_disruption_closure",
            list(path) + list(item.get("dependency_path", [])),
            accepted, rejected,
        )
    return accepted, rejected


def _extract_relation_content_closure(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    accepted = []
    rejected = []
    for item in _depth_one_relations(
        base, POSITIONAL_RELATIONS | DIRECTION_RELATIONS
    ):
        child = _relation_operand(item)
        child_ids = v24._operand_entity_ids(child)
        child_tokens = v23._operand_token_ids(child)
        if not child_tokens:
            continue
        child_end = max(child_tokens)
        candidates = [
            entity for entity in base["entities"]
            if entity.get("entity_id") not in child_ids
            and entity.get("head_token_id") is not None
            and int(entity["token_span"][0]) + 1 > child_end
        ]
        if len(candidates) != 1:
            continue
        content = candidates[0]
        content_start = int(content["token_span"][0]) + 1
        markers = [
            (token_id, tokens[token_id - 1])
            for token_id in range(child_end + 1, content_start)
            if tokens[token_id - 1] in OCCURRENCE_MARKERS
        ]
        if len(markers) != 1:
            continue
        marker_id, marker = markers[0]
        clause_start = marker_id
        while clause_start > 1 and tokens[clause_start - 2] not in v22.CLAUSE_PUNCTUATION:
            clause_start -= 1
        if any(
            tokens[token_id - 1] in {"位于", "处于", "分布", "分布于"}
            for token_id in range(clause_start, marker_id)
        ):
            continue
        path = _path_between(
            item["trigger_token_id"], int(content["head_token_id"]),
            heads, dependency_relations, max_steps=10,
        )
        if path is None:
            continue
        _accept(
            type_schema, "IN", marker, marker_id, child, content,
            "explicit_relation_content_closure",
            list(path) + list(item.get("dependency_path", [])),
            accepted, rejected,
        )
    return accepted, rejected


def _extract_outer_scope_closure(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    accepted = []
    rejected = []
    for item in _depth_one_relations(base, POSITIONAL_RELATIONS):
        child = _relation_operand(item)
        child_ids = v24._operand_entity_ids(child)
        child_start = min(v23._operand_token_ids(child))
        outers = [
            entity for entity in base["entities"]
            if entity.get("entity_id") not in child_ids
            and entity.get("head_token_id") is not None
            and int(entity["token_span"][1]) + 1 < child_start
        ]
        if not outers:
            continue
        nearest_end = max(int(entity["token_span"][1]) + 1 for entity in outers)
        outers = [
            entity for entity in outers
            if int(entity["token_span"][1]) + 1 == nearest_end
        ]
        if len(outers) != 1:
            continue
        outer = outers[0]
        outer_end = int(outer["token_span"][1]) + 1
        markers = [
            (token_id, tokens[token_id - 1])
            for token_id in range(outer_end + 1, child_start)
            if tokens[token_id - 1] in SCOPE_MARKERS
        ]
        if len(markers) != 1:
            continue
        marker_id, marker = markers[0]
        clause_start = marker_id
        while clause_start > 1 and tokens[clause_start - 2] not in v22.CLAUSE_PUNCTUATION:
            clause_start -= 1
        if any(
            tokens[token_id - 1] in {"位于", "处于", "分布", "分布于"}
            for token_id in range(clause_start, marker_id)
        ):
            continue
        path = _path_between(
            int(outer["head_token_id"]), item["trigger_token_id"],
            heads, dependency_relations, max_steps=10,
        )
        if path is None:
            continue
        _accept(
            type_schema, "IN", marker, marker_id, outer, child,
            "explicit_outer_positional_scope_closure",
            list(path) + list(item.get("dependency_path", [])),
            accepted, rejected,
        )
    return accepted, rejected


def _extract_positional_identity_closure(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    accepted = []
    rejected = []
    for item in _depth_one_relations(base, POSITIONAL_RELATIONS):
        child = _relation_operand(item)
        child_ids = v24._operand_entity_ids(child)
        child_end = max(v23._operand_token_ids(child))
        targets = [
            entity for entity in base["entities"]
            if entity.get("entity_id") not in child_ids
            and entity.get("head_token_id") is not None
            and int(entity["token_span"][0]) + 1 > child_end
        ]
        if len(targets) != 1:
            continue
        target = targets[0]
        target_start = int(target["token_span"][0]) + 1
        markers = [
            (token_id, tokens[token_id - 1])
            for token_id in range(child_end + 1, target_start)
            if tokens[token_id - 1] in IDENTITY_MARKERS
        ]
        if len(markers) != 1:
            continue
        marker_id, marker = markers[0]
        path = _path_between(
            item["trigger_token_id"], int(target["head_token_id"]),
            heads, dependency_relations, max_steps=10,
        )
        if path is None:
            continue
        _accept(
            type_schema, "EQ", marker, marker_id, child, target,
            "explicit_positional_identity_closure",
            list(path) + list(item.get("dependency_path", [])),
            accepted, rejected,
        )
    return accepted, rejected


def extract_relation_evidence(row, type_schema, enable_v27_closure=True):
    base = v27.extract_relation_evidence(
        row, type_schema, enable_explicit_l1=True
    )
    if not enable_v27_closure:
        result = dict(base)
        result["version"] = "v28_compatibility"
        return result

    parent_added, parent_rejections = v26._extract_child_parents(
        row, base, type_schema
    )
    closure_added = []
    closure_rejections = []
    for extractor in (
        _extract_contact_closure,
        _extract_disruption_closure,
        _extract_relation_content_closure,
        _extract_outer_scope_closure,
        _extract_positional_identity_closure,
    ):
        added, rejected = extractor(row, base, type_schema)
        closure_added.extend(added)
        closure_rejections.extend(rejected)
    parent_added = list(parent_added) + closure_added
    parent_rejections = list(parent_rejections) + closure_rejections
    clause_ids = v23._clause_ids(list(row.get("tokens", [])))
    annotated_parents = [
        item if item.get("node") else v23._annotate_evidence(item, clause_ids)
        for item in parent_added
    ]
    scope_added, scope_rejections = v26._extract_explicit_hierarchical_scopes(
        row, base, annotated_parents, type_schema
    )

    accepted = []
    seen = set()
    for item in list(base["accepted"]) + annotated_parents + scope_added:
        key = (
            item["relation"], v22._operand_key(item["left"]),
            v22._operand_key(item["right"]),
        )
        if key in seen:
            continue
        seen.add(key)
        accepted.append(
            item if item.get("node") else v23._annotate_evidence(item, clause_ids)
        )

    result = dict(base)
    result["accepted"] = accepted
    result["rejected"] = (
        list(base["rejected"]) + parent_rejections + scope_rejections
    )
    result["version"] = "v28"
    return result
