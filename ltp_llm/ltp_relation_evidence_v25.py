#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V25 explicit L1 overlap and between evidence.

V25 keeps V24 unchanged.  It adds only surface-explicit PO/DC structures and
then reruns V24's existing bottom-up wrappers over the enlarged accepted set.
No default relation, arbitrary entity-pair enumeration, generic COO
propagation, or unrestricted ATT propagation is used.
"""

from __future__ import print_function

try:
    from . import ltp_relation_evidence_v22 as v22
    from . import ltp_relation_evidence_v23 as v23
    from . import ltp_relation_evidence_v24 as v24
except (ImportError, ValueError):
    import ltp_relation_evidence_v22 as v22
    import ltp_relation_evidence_v23 as v23
    import ltp_relation_evidence_v24 as v24


INFIX_OVERLAP_MARKERS = {"夹", "夹有", "夹含", "夹杂"}
COORDINATED_OVERLAP_MARKERS = {"互层", "间互", "交替"}
COORDINATORS = {"与", "和"}
EXPLICIT_PATH_RELATIONS = {
    "ATT", "SBV", "VOB", "POB", "ADV", "CMP", "COO", "LAD",
}

load_type_schema = v24.load_type_schema
schema_allows = v24.schema_allows


def _entity_start(entity):
    return int(entity["token_span"][0]) + 1


def _entity_end(entity):
    return int(entity["token_span"][1]) + 1


def _same_clause_entities(entities, token_id, clause_ids):
    clause_id = clause_ids[token_id - 1]
    return [
        entity for entity in entities
        if entity.get("head_token_id") is not None
        and clause_ids[int(entity["head_token_id"]) - 1] == clause_id
    ]


def _nearest_before(entities, token_id):
    values = [entity for entity in entities if _entity_end(entity) < token_id]
    if not values:
        return []
    distance = min(token_id - _entity_end(entity) for entity in values)
    return [
        entity for entity in values
        if token_id - _entity_end(entity) == distance
    ]


def _nearest_after(entities, token_id):
    values = [entity for entity in entities if _entity_start(entity) > token_id]
    if not values:
        return []
    distance = min(_entity_start(entity) - token_id for entity in values)
    return [
        entity for entity in values
        if _entity_start(entity) - token_id == distance
    ]


def _explicit_path(entity, trigger_id, heads, dependency_relations,
                   max_steps=4):
    return v24._dependency_path_between(
        int(entity["head_token_id"]), trigger_id,
        heads, dependency_relations, EXPLICIT_PATH_RELATIONS,
        max_steps=max_steps,
    )


def _tokens_between(tokens, left, right):
    return tokens[_entity_end(left):_entity_start(right) - 1]


def _coordinated_pair_before(tokens, entities, trigger_id, heads,
                             dependency_relations):
    before = [entity for entity in entities if _entity_end(entity) < trigger_id]
    before.sort(key=lambda item: (_entity_end(item), _entity_start(item)))
    if len(before) < 2:
        return None
    left, right = before[-2], before[-1]
    between = _tokens_between(tokens, left, right)
    if len([token for token in between if token in COORDINATORS]) != 1:
        return None
    pair_path = v24._dependency_path_between(
        int(left["head_token_id"]), int(right["head_token_id"]),
        heads, dependency_relations, {"COO", "LAD", "ATT"},
        max_steps=2,
    )
    if pair_path is None or not any(
        edge.get("relation") in {"COO", "LAD"} for edge in pair_path
    ):
        return None
    left_path = _explicit_path(
        left, trigger_id, heads, dependency_relations, max_steps=4
    )
    right_path = _explicit_path(
        right, trigger_id, heads, dependency_relations, max_steps=4
    )
    if left_path is None or right_path is None:
        return None
    return left, right, pair_path + left_path + right_path


def _extract_infix_overlap(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []
    for trigger_id, lexeme in enumerate(tokens, 1):
        if lexeme not in INFIX_OVERLAP_MARKERS:
            continue
        entities = _same_clause_entities(base["entities"], trigger_id, clause_ids)
        left_values = _nearest_before(entities, trigger_id)
        right_values = _nearest_after(entities, trigger_id)
        if len(left_values) != 1 or len(right_values) != 1:
            rejected.append(v22._rejection(
                lexeme, trigger_id, "explicit_infix_overlap",
                "explicit_overlap_arguments_not_unique",
            ))
            continue
        left, right = left_values[0], right_values[0]
        if left.get("type") == "SM" and not left.get("text", "").endswith("岩组"):
            rejected.append(v22._rejection(
                lexeme, trigger_id, "explicit_infix_overlap",
                "left_operand_is_non_lithologic_stratigraphic_unit",
                {"left": left["entity_id"], "text": left.get("text", "")},
            ))
            continue
        left_path = _explicit_path(
            left, trigger_id, heads, dependency_relations, max_steps=4
        )
        right_path = _explicit_path(
            right, trigger_id, heads, dependency_relations, max_steps=4
        )
        if left_path is None or right_path is None:
            rejected.append(v22._rejection(
                lexeme, trigger_id, "explicit_infix_overlap",
                "explicit_overlap_dependency_path_missing",
            ))
            continue
        evidence = v22._evidence_record(
            "PO", lexeme, trigger_id, left, right,
            "explicit_infix_overlap", left_path + right_path,
        )
        v22._accept_if_valid(type_schema, evidence, accepted, rejected)
    return accepted, rejected


def _extract_coordinated_overlap(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []
    for trigger_id, lexeme in enumerate(tokens, 1):
        if lexeme not in COORDINATED_OVERLAP_MARKERS:
            continue
        entities = _same_clause_entities(base["entities"], trigger_id, clause_ids)
        pair = _coordinated_pair_before(
            tokens, entities, trigger_id, heads, dependency_relations
        )
        if pair is None:
            rejected.append(v22._rejection(
                lexeme, trigger_id, "explicit_coordinated_overlap",
                "explicit_coordinated_overlap_pair_missing",
            ))
            continue
        left, right, path = pair
        evidence = v22._evidence_record(
            "PO", lexeme, trigger_id, left, right,
            "explicit_coordinated_overlap", path,
        )
        v22._accept_if_valid(type_schema, evidence, accepted, rejected)
    return accepted, rejected


def _extract_between(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []
    for trigger_id, lexeme in enumerate(tokens, 1):
        if lexeme != "之间":
            continue
        entities = _same_clause_entities(base["entities"], trigger_id, clause_ids)
        pair = _coordinated_pair_before(
            tokens, entities, trigger_id, heads, dependency_relations
        )
        if pair is None:
            rejected.append(v22._rejection(
                lexeme, trigger_id, "explicit_between",
                "explicit_between_pair_missing",
            ))
            continue
        left, right, path = pair
        evidence = v22._evidence_record(
            "DC", lexeme, trigger_id, left, right,
            "explicit_coordinated_between", path,
        )
        v22._accept_if_valid(type_schema, evidence, accepted, rejected)
    return accepted, rejected


def _extract_between_parents(row, base, between_items, type_schema):
    """Build ``IN(DC(A, B), X)`` when X is the unique linked third entity."""
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    accepted = []
    rejected = []
    for item in between_items:
        pair_ids = {
            item["left"]["entity_id"], item["right"]["entity_id"],
        }
        targets = [
            entity for entity in base["entities"]
            if entity["entity_id"] not in pair_ids
            and entity.get("head_token_id") is not None
        ]
        if len(targets) != 1:
            rejected.append(v22._rejection(
                item["lexeme"], item["trigger_token_id"],
                "explicit_between_parent",
                "explicit_between_parent_not_unique",
                {"targets": [entity["entity_id"] for entity in targets]},
            ))
            continue
        target = targets[0]
        target_path = v24._dependency_path_between(
            int(target["head_token_id"]), int(item["trigger_token_id"]),
            heads, dependency_relations,
            EXPLICIT_PATH_RELATIONS | {"FOB"}, max_steps=8,
        )
        if target_path is None or not target_path:
            rejected.append(v22._rejection(
                item["lexeme"], item["trigger_token_id"],
                "explicit_between_parent",
                "explicit_between_parent_dependency_path_missing",
            ))
            continue
        dc_operand = v22._relation_operand(
            item["relation"], item["lexeme"], item["trigger_token_id"],
            item["left"], item["right"], item["pattern"],
            item.get("dependency_path", []),
        )
        evidence = v22._evidence_record(
            "IN", item["lexeme"], item["trigger_token_id"],
            dc_operand, target, "explicit_between_parent",
            list(item.get("dependency_path", [])) + target_path,
        )
        v22._accept_if_valid(type_schema, evidence, accepted, rejected)
    return accepted, rejected


def _deduplicate(items):
    result = []
    seen = set()
    for item in items:
        key = (
            item["relation"], v22._operand_key(item["left"]),
            v22._operand_key(item["right"]),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def extract_relation_evidence(row, type_schema, enable_explicit_l1=True):
    base = v24.extract_relation_evidence(
        row, type_schema,
        enable_positional_wrapper=True,
        enable_carrier_wrapper=True,
    )
    if not enable_explicit_l1:
        result = dict(base)
        result["version"] = "v25_compatibility"
        return result

    infix, infix_rejections = _extract_infix_overlap(row, base, type_schema)
    coordinated, coordinated_rejections = _extract_coordinated_overlap(
        row, base, type_schema
    )
    between, between_rejections = _extract_between(row, base, type_schema)
    between_parents, between_parent_rejections = _extract_between_parents(
        row, base, between, type_schema
    )
    explicit_l1 = _deduplicate(infix + coordinated + between)

    # Feed the new L1 nodes back through V24's already audited bottom-up
    # wrappers.  This is what lets one explicit PO unlock L2/L3.
    working = dict(base)
    working["accepted"] = _deduplicate(list(base["accepted"]) + explicit_l1)
    positional, positional_rejections = v24._extract_positional_wrappers(
        row, working, type_schema
    )
    carrier_working = dict(working)
    carrier_working["accepted"] = _deduplicate(
        list(working["accepted"]) + positional
    )
    carriers, carrier_rejections = v24._extract_carrier_wrappers(
        row, carrier_working, type_schema
    )
    outer, outer_rejections = v24._extract_outer_scopes(
        row, working, positional + carriers, type_schema
    )

    clause_ids = v23._clause_ids(list(row.get("tokens", [])))
    accepted = []
    seen = set()
    for item in (
        list(base["accepted"]) + explicit_l1 + between_parents
        + positional + carriers + outer
    ):
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
        list(base["rejected"]) + infix_rejections
        + coordinated_rejections + between_rejections
        + between_parent_rejections
        + positional_rejections + carrier_rejections + outer_rejections
    )
    result["version"] = "v25"
    return result
