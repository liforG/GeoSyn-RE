#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conservative interpretation of GNRED LTP dependency evidence.

This module deliberately does not import ``nested_pipeline``.  It consumes the
already enriched JSONL rows produced by ``run_batch_2.py`` and turns direct LTP
dependencies into auditable relation evidence.  Unknown or incomplete
structures are rejected instead of being converted to ``IN``.
"""

from __future__ import print_function

import collections
import json


# Complete inventory inherited from the current pipeline.  Inclusion here does
# not imply that an occurrence is accepted: every family below has an explicit
# syntactic contract, and unsupported constructions are reported as rejected.
RELATION_LEXICON = {
    "EQ": ("是", "为", "属于", "属"),
    "IN": (
        "含", "包括", "位于", "在", "由", "构成", "组成", "包含",
        "分布", "处于", "出现", "出现于", "出现在", "中", "之中",
        "内部", "内",
    ),
    "PO": (
        "伴", "交错", "穿插", "穿越", "叠置", "交汇", "相交", "夹杂",
        "夹", "夹有", "互层",
    ),
    "A": ("上", "上部", "之上", "顶部"),
    "BW": ("下", "下部", "之下", "底部", "下伏", "下段", "下亚段"),
    "C": ("中部", "中间", "轴部", "中段"),
    "N": ("北", "北部", "以北", "北侧", "北缘"),
    "S": ("南", "南部", "以南", "南侧", "南缘"),
    "E": ("东", "东部", "以东", "东侧", "东缘"),
    "W": ("西", "西部", "以西", "西侧", "西缘"),
    "NE": ("东北", "东北部", "东北侧"),
    "SE": ("东南", "东南部", "东南侧"),
    "NW": ("西北", "西北部", "西北侧"),
    "SW": ("西南", "西南部", "西南侧"),
}

LEXEME_TO_RELATION = {
    lexeme: relation
    for relation, lexemes in RELATION_LEXICON.items()
    for lexeme in lexemes
}

DIRECT_EQ = set(RELATION_LEXICON["EQ"])
DIRECT_CONTAINMENT = {"含", "包括", "包含"}
DIRECT_OVERLAP = set(RELATION_LEXICON["PO"])
LOCATIVE_PREDICATES = {"位于", "分布", "处于", "出现", "出现于", "出现在"}
DEFERRED_COMPOSITION = {"由", "构成", "组成"}
AMBIGUOUS_STANDALONE = {"在", "中", "之中", "内部", "内"}
DEFERRED_DIRECT_PREDICATES = {
    "为", "属于", "属", "含", "位于", "分布", "处于", "出现", "出现于", "出现在",
}
DEFERRED_GENERIC_LOCALIZERS = {"中", "之中", "内部", "内"}

LOCALIZER_RELATIONS = {
    lexeme: relation
    for relation in ("IN", "A", "BW", "C", "N", "S", "E", "W", "NE", "SE", "NW", "SW")
    for lexeme in RELATION_LEXICON[relation]
    if lexeme not in DIRECT_CONTAINMENT
    and lexeme not in LOCATIVE_PREDICATES
    and lexeme not in DEFERRED_COMPOSITION
    and lexeme != "在"
}

CLAUSE_PUNCTUATION = {"，", "。", "；", "：", ",", ";", ":", "!", "！", "?", "？"}


def load_type_schema(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return {
        str(relation): list(rules)
        for relation, rules in data.get("relation", [])
    }


def schema_allows(type_schema, relation, left_type, right_type):
    for left_allowed, right_allowed in type_schema.get(str(relation), []):
        if left_type in left_allowed and right_type in right_allowed:
            return True
    return False


def _dependency_maps(row):
    tokens = list(row.get("tokens", []))
    heads = {}
    relations = {}
    children = collections.defaultdict(list)
    for item in row.get("dep", []):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        dependent = int(item[0])
        head = int(item[1])
        relation = str(item[2])
        if dependent < 1 or dependent > len(tokens) or head < 0 or head > len(tokens):
            continue
        heads[dependent] = head
        relations[dependent] = relation
        children[head].append((dependent, relation))
    for values in children.values():
        values.sort(key=lambda item: item[0])
    return heads, relations, children


def _entity_token_spans(row):
    """Recover one stable token span for every input entity index."""
    entities = list(row.get("entities", []))
    candidates = collections.defaultdict(set)
    for values in (row.get("entity_map") or {}).values():
        if not isinstance(values, list):
            continue
        for mapped in values:
            if not isinstance(mapped, dict):
                continue
            span = mapped.get("span")
            token_span = mapped.get("token_span")
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            if not isinstance(token_span, (list, tuple)) or len(token_span) != 2:
                continue
            key = (
                str(mapped.get("text", "")),
                int(span[0]),
                int(span[1]),
                str(mapped.get("type", "")),
            )
            candidates[key].add((int(token_span[0]), int(token_span[1])))

    result = []
    for entity_index, entity in enumerate(entities):
        span = entity.get("span") or []
        key = (
            str(entity.get("text", "")),
            int(span[0]) if len(span) == 2 else -1,
            int(span[1]) if len(span) == 2 else -1,
            str(entity.get("type", "")),
        )
        spans = sorted(candidates.get(key, []))
        result.append(spans[0] if len(spans) == 1 else None)
    return result


def build_entities(row, heads):
    token_spans = _entity_token_spans(row)
    records = []
    for entity_index, raw in enumerate(row.get("entities", [])):
        token_span = token_spans[entity_index] if entity_index < len(token_spans) else None
        head_token_id = None
        head_status = "missing_token_span"
        if token_span is not None:
            start, end = token_span
            token_ids = set(range(start + 1, end + 2))
            external = [token_id for token_id in sorted(token_ids) if heads.get(token_id, 0) not in token_ids]
            if len(external) == 1:
                head_token_id = external[0]
                head_status = "unique_external_head"
            elif len(token_ids) == 1:
                head_token_id = next(iter(token_ids))
                head_status = "single_token"
            elif not external:
                head_status = "no_external_head"
            else:
                head_status = "multiple_external_heads"

        records.append({
            "kind": "entity",
            "entity_id": "e{}".format(entity_index),
            "entity_index": entity_index,
            "text": str(raw.get("text", "")),
            "type": str(raw.get("type", "")),
            "span": list(raw.get("span", [])),
            "token_span": list(token_span) if token_span is not None else None,
            "head_token_id": head_token_id,
            "head_status": head_status,
        })
    return records


def _entity_index_by_head(entities):
    result = collections.defaultdict(list)
    for entity in entities:
        if entity.get("head_token_id") is not None:
            result[int(entity["head_token_id"])].append(entity)
    return result


def _direct_entities(children, entities_by_head, head_token_id, relations):
    result = []
    for dependent, dependency_relation in children.get(head_token_id, []):
        if dependency_relation not in relations:
            continue
        result.extend(entities_by_head.get(dependent, []))
    return _unique_entities(result)


def _unique_entities(values):
    result = []
    seen = set()
    for value in values:
        key = value.get("entity_id")
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _prepositional_objects(tokens, pos, children, entities_by_head, predicate_id):
    result = []
    used_prepositions = []
    for child_id, dependency_relation in children.get(predicate_id, []):
        if dependency_relation not in {"ADV", "CMP"}:
            continue
        child_pos = pos[child_id - 1] if child_id - 1 < len(pos) else ""
        child_text = tokens[child_id - 1] if child_id - 1 < len(tokens) else ""
        if child_pos != "p" and child_text not in {"于", "在", "向", "往"}:
            continue
        objects = _direct_entities(children, entities_by_head, child_id, {"POB"})
        if objects:
            result.extend(objects)
            used_prepositions.append({"token_id": child_id, "text": child_text})
    return _unique_entities(result), used_prepositions


def _has_localizer_attachment(tokens, children, predicate_id):
    for dependent, relation in children.get(predicate_id, []):
        if relation in {"VOB", "POB", "SBV", "ADV"} and tokens[dependent - 1] in LOCALIZER_RELATIONS:
            return True
    return False


def _has_descendant_localizer(tokens, children, predicate_id):
    queue = [predicate_id]
    seen = {predicate_id}
    while queue:
        current = queue.pop(0)
        for dependent, relation in children.get(current, []):
            if dependent in seen or relation in {"WP", "RAD"}:
                continue
            seen.add(dependent)
            if tokens[dependent - 1] in LOCALIZER_RELATIONS:
                return True
            queue.append(dependent)
    return False


def _operand_key(operand):
    if operand.get("kind") == "relation":
        return (
            "relation", operand["relation"],
            _operand_key(operand["left"]), _operand_key(operand["right"]),
        )
    return ("entity", operand.get("entity_id"))


def _operand_type(operand):
    if operand.get("kind") == "relation":
        return str(operand["relation"])
    return str(operand.get("type", ""))


def _relation_operand(relation, lexeme, trigger_id, left, right, pattern, path):
    return {
        "kind": "relation",
        "relation": relation,
        "lexeme": lexeme,
        "trigger_token_id": trigger_id,
        "left": left,
        "right": right,
        "pattern": pattern,
        "dependency_path": path,
    }


def _evidence_record(relation, lexeme, trigger_id, left, right, pattern, path):
    return {
        "relation": relation,
        "lexeme": lexeme,
        "trigger_token_id": trigger_id,
        "left": left,
        "right": right,
        "pattern": pattern,
        "dependency_path": path,
    }


def _rejection(lexeme, trigger_id, family, reason, details=None):
    record = {
        "lexeme": lexeme,
        "trigger_token_id": trigger_id,
        "family": family,
        "reason": reason,
    }
    if details:
        record["details"] = details
    return record


def _accept_if_valid(type_schema, evidence, accepted, rejected):
    left = evidence["left"]
    right = evidence["right"]
    if _operand_key(left) == _operand_key(right):
        rejected.append(_rejection(
            evidence["lexeme"], evidence["trigger_token_id"],
            evidence["pattern"], "same_entity_operands",
        ))
        return
    left_type = _operand_type(left)
    right_type = _operand_type(right)
    if not schema_allows(type_schema, evidence["relation"], left_type, right_type):
        rejected.append(_rejection(
            evidence["lexeme"], evidence["trigger_token_id"],
            evidence["pattern"], "schema_disallowed",
            {"relation": evidence["relation"], "left_type": left_type, "right_type": right_type},
        ))
        return
    accepted.append(evidence)


def _extract_direct_predicates(row, type_schema, entities, heads, dependency_relations, children):
    tokens = list(row.get("tokens", []))
    pos = list(row.get("pos", []))
    entities_by_head = _entity_index_by_head(entities)
    accepted = []
    rejected = []

    for token_index, lexeme in enumerate(tokens):
        trigger_id = token_index + 1
        if lexeme not in LEXEME_TO_RELATION:
            continue
        if lexeme in LOCALIZER_RELATIONS:
            continue
        if lexeme in DEFERRED_COMPOSITION:
            rejected.append(_rejection(lexeme, trigger_id, "composition", "deferred_composition_rule"))
            continue
        if lexeme in AMBIGUOUS_STANDALONE:
            rejected.append(_rejection(lexeme, trigger_id, "ambiguous", "standalone_ambiguous_lexeme"))
            continue
        if lexeme in DEFERRED_DIRECT_PREDICATES:
            rejected.append(_rejection(
                lexeme, trigger_id, "direct_predicate", "deferred_ambiguous_direct_semantics"
            ))
            continue
        if lexeme not in DIRECT_EQ and lexeme not in DIRECT_CONTAINMENT and lexeme not in DIRECT_OVERLAP and lexeme not in LOCATIVE_PREDICATES:
            rejected.append(_rejection(lexeme, trigger_id, "unsupported", "no_approved_direct_pattern"))
            continue

        subjects = _direct_entities(children, entities_by_head, trigger_id, {"SBV"})
        direct_objects = _direct_entities(children, entities_by_head, trigger_id, {"VOB"})
        prep_objects, prepositions = _prepositional_objects(
            tokens, pos, children, entities_by_head, trigger_id
        )
        objects = _unique_entities(direct_objects + prep_objects)

        if len(subjects) != 1 or len(objects) != 1:
            rejected.append(_rejection(
                lexeme, trigger_id, "direct_predicate", "incomplete_or_ambiguous_arguments",
                {"subjects": [item["entity_id"] for item in subjects],
                 "objects": [item["entity_id"] for item in objects]},
            ))
            continue

        subject = subjects[0]
        obj = objects[0]

        if lexeme in DIRECT_OVERLAP and (subject["type"] != "RK" or obj["type"] != "RK"):
            rejected.append(_rejection(
                lexeme, trigger_id, "direct_predicate", "deferred_non_rock_overlap_operands",
                {"subject_type": subject["type"], "object_type": obj["type"]},
            ))
            continue

        if lexeme in DIRECT_CONTAINMENT and any(
            dependency_relation == "COO" and tokens[dependent - 1] == "夹"
            for dependent, dependency_relation in children.get(trigger_id, [])
        ):
            rejected.append(_rejection(
                lexeme, trigger_id, "direct_predicate", "nested_overlap_requires_composition"
            ))
            continue
        path = [
            {"dependent": subject["head_token_id"], "head": trigger_id, "relation": "SBV"},
        ]
        if direct_objects:
            path.append({"dependent": obj["head_token_id"], "head": trigger_id, "relation": "VOB"})
        else:
            path.extend({"preposition": item} for item in prepositions)
            path.append({"dependent": obj["head_token_id"], "relation": "POB"})

        if lexeme in LOCATIVE_PREDICATES:
            if _has_localizer_attachment(tokens, children, trigger_id) or _has_descendant_localizer(tokens, children, trigger_id):
                rejected.append(_rejection(
                    lexeme, trigger_id, "locative_predicate", "localizer_requires_nested_interpretation"
                ))
                continue
            if lexeme == "出现" and not prep_objects:
                rejected.append(_rejection(
                    lexeme, trigger_id, "locative_predicate", "bare_occurrence_is_not_locative_enough"
                ))
                continue
            evidence = _evidence_record(
                "IN", lexeme, trigger_id, obj, subject,
                "locative_sbv_object_inverted", path,
            )
        else:
            relation = LEXEME_TO_RELATION[lexeme]
            evidence = _evidence_record(
                relation, lexeme, trigger_id, subject, obj,
                "direct_sbv_object", path,
            )
        _accept_if_valid(type_schema, evidence, accepted, rejected)

    return accepted, rejected


def _preposition_child_objects(tokens, pos, children, entities_by_head, predicate_id, lexeme):
    result = []
    preposition_ids = []
    for child_id, dependency_relation in children.get(predicate_id, []):
        if dependency_relation not in {"ADV", "CMP"}:
            continue
        if tokens[child_id - 1] != lexeme:
            continue
        objects = _direct_entities(children, entities_by_head, child_id, {"POB"})
        result.extend(objects)
        if objects:
            preposition_ids.append(child_id)
    return _unique_entities(result), preposition_ids


def _extract_nested_overlap(row, type_schema, entities, heads, dependency_relations, children):
    """Handle dependency-closed ``X [localizer] 为 A 夹 B``/``X 含 A 夹 B``.

    This is not general COO propagation.  The rule is accepted only when the
    inner ``夹`` operands and the outer operand are uniquely identified by the
    direct dependency edges described below.
    """
    tokens = list(row.get("tokens", []))
    pos = list(row.get("pos", []))
    entities_by_head = _entity_index_by_head(entities)
    accepted = []
    rejected = []

    for token_index, lexeme in enumerate(tokens):
        if lexeme != "夹":
            continue
        trigger_id = token_index + 1
        right_values = _direct_entities(children, entities_by_head, trigger_id, {"VOB"})
        direct_subjects = _direct_entities(children, entities_by_head, trigger_id, {"SBV"})
        for_values, for_ids = _preposition_child_objects(
            tokens, pos, children, entities_by_head, trigger_id, "为"
        )

        outer_predicate_id = heads.get(trigger_id, 0) if dependency_relations.get(trigger_id) == "COO" else 0
        outer_lexeme = tokens[outer_predicate_id - 1] if outer_predicate_id > 0 else ""
        outer_objects = []
        if outer_lexeme == "含":
            outer_objects = _direct_entities(children, entities_by_head, outer_predicate_id, {"VOB"})

        left_values = for_values if for_values else outer_objects
        if len(left_values) != 1 or len(right_values) != 1:
            continue

        inner = _relation_operand(
            "PO", "夹", trigger_id, left_values[0], right_values[0],
            "nested_overlap_inner",
            [
                {"dependent": left_values[0]["head_token_id"], "relation": "POB_or_outer_VOB"},
                {"dependent": right_values[0]["head_token_id"], "head": trigger_id, "relation": "VOB"},
            ],
        )
        inner_evidence = _evidence_record(
            "PO", "夹", trigger_id, left_values[0], right_values[0],
            "nested_overlap_inner", inner["dependency_path"],
        )
        inner_accepted = []
        _accept_if_valid(type_schema, inner_evidence, inner_accepted, rejected)
        if not inner_accepted:
            continue
        accepted.extend(inner_accepted)

        outer_evidence = None
        if for_values:
            if len(direct_subjects) == 1:
                rejected.append(_rejection(
                    "为…夹", trigger_id, "nested_overlap",
                    "deferred_ambiguous_copular_outer_relation",
                    {"subject": direct_subjects[0]["entity_id"], "inner_relation": "PO"},
                ))
            else:
                localizer_subjects = [
                    dependent for dependent, relation in children.get(trigger_id, [])
                    if relation == "SBV" and tokens[dependent - 1] in LOCALIZER_RELATIONS
                ]
                if len(localizer_subjects) == 1:
                    localizer_id = localizer_subjects[0]
                    anchors = _direct_entities(children, entities_by_head, localizer_id, {"ATT"})
                    if len(anchors) == 1:
                        outer_evidence = _evidence_record(
                            LOCALIZER_RELATIONS[tokens[localizer_id - 1]],
                            tokens[localizer_id - 1], localizer_id, anchors[0], inner,
                            "localizer_nested_overlap",
                            [
                                {"dependent": anchors[0]["head_token_id"], "head": localizer_id, "relation": "ATT"},
                                {"dependent": localizer_id, "head": trigger_id, "relation": "SBV"},
                            ],
                        )
        elif outer_lexeme == "含":
            outer_subjects = _direct_entities(children, entities_by_head, outer_predicate_id, {"SBV"})
            if len(outer_subjects) == 1:
                outer_evidence = _evidence_record(
                    "IN", "含…夹", outer_predicate_id, outer_subjects[0], inner,
                    "containment_nested_overlap",
                    [
                        {"dependent": outer_subjects[0]["head_token_id"], "head": outer_predicate_id, "relation": "SBV"},
                        {"dependent": trigger_id, "head": outer_predicate_id, "relation": "COO"},
                    ],
                )

        if outer_evidence is not None:
            _accept_if_valid(type_schema, outer_evidence, accepted, rejected)

    return accepted, rejected


def _predicate_for_localizer(heads, dependency_relations, token_id):
    relation = dependency_relations.get(token_id)
    head = heads.get(token_id, 0)
    if relation in {"SBV", "VOB", "ADV"} and head > 0:
        return head, relation, None
    if relation == "POB" and head > 0:
        preposition_id = head
        predicate_id = heads.get(preposition_id, 0)
        if predicate_id > 0 and dependency_relations.get(preposition_id) in {"ADV", "CMP"}:
            return predicate_id, relation, preposition_id
    return None, relation, None


def _extract_localizers(row, type_schema, entities, heads, dependency_relations, children):
    tokens = list(row.get("tokens", []))
    entities_by_head = _entity_index_by_head(entities)
    accepted = []
    rejected = []

    for token_index, lexeme in enumerate(tokens):
        if lexeme not in LOCALIZER_RELATIONS:
            continue
        trigger_id = token_index + 1
        if lexeme in DEFERRED_GENERIC_LOCALIZERS:
            rejected.append(_rejection(
                lexeme, trigger_id, "localizer", "deferred_generic_containment_localizer"
            ))
            continue
        relation = LOCALIZER_RELATIONS[lexeme]
        anchors = _direct_entities(children, entities_by_head, trigger_id, {"ATT"})
        predicate_id, localizer_role, preposition_id = _predicate_for_localizer(
            heads, dependency_relations, trigger_id
        )
        if predicate_id is None:
            rejected.append(_rejection(
                lexeme, trigger_id, "localizer", "no_direct_governing_predicate"
            ))
            continue

        if tokens[predicate_id - 1] == "夹" and any(
            dependency_relation in {"ADV", "CMP"} and tokens[dependent - 1] == "为"
            for dependent, dependency_relation in children.get(predicate_id, [])
        ):
            rejected.append(_rejection(
                lexeme, trigger_id, "localizer", "nested_overlap_requires_composition"
            ))
            continue

        if localizer_role == "POB":
            outer_subjects = _direct_entities(children, entities_by_head, predicate_id, {"SBV"})
            intermediate_targets = _direct_entities(children, entities_by_head, predicate_id, {"VOB"})
            if dependency_relations.get(predicate_id) == "ATT":
                intermediate_targets = _unique_entities(
                    intermediate_targets + entities_by_head.get(heads.get(predicate_id, 0), [])
                )
            if intermediate_targets:
                if len(anchors) != 1 or len(intermediate_targets) != 1 or len(outer_subjects) != 1:
                    rejected.append(_rejection(
                        lexeme, trigger_id, "localizer", "ambiguous_nested_locative_arguments",
                        {"anchors": [item["entity_id"] for item in anchors],
                         "intermediate_targets": [item["entity_id"] for item in intermediate_targets],
                         "outer_subjects": [item["entity_id"] for item in outer_subjects]},
                    ))
                    continue

                inner_path = [
                    {"dependent": anchors[0]["head_token_id"], "head": trigger_id, "relation": "ATT"},
                    {"dependent": trigger_id, "head": heads.get(trigger_id), "relation": "POB"},
                    {"dependent": intermediate_targets[0]["head_token_id"],
                     "head": predicate_id, "relation": "VOB_or_predicate_ATT_head"},
                ]
                inner = _relation_operand(
                    relation, lexeme, trigger_id, anchors[0], intermediate_targets[0],
                    "nested_locative_inner", inner_path,
                )
                inner_evidence = _evidence_record(
                    relation, lexeme, trigger_id, anchors[0], intermediate_targets[0],
                    "nested_locative_inner", inner_path,
                )
                inner_accepted = []
                _accept_if_valid(type_schema, inner_evidence, inner_accepted, rejected)
                if not inner_accepted:
                    continue
                accepted.extend(inner_accepted)

                outer_path = [
                    {"dependent": outer_subjects[0]["head_token_id"], "head": predicate_id, "relation": "SBV"},
                    {"dependent": trigger_id, "relation": "locative_POB"},
                ]
                outer_evidence = _evidence_record(
                    "IN", tokens[predicate_id - 1], predicate_id, inner, outer_subjects[0],
                    "nested_locative_outer", outer_path,
                )
                _accept_if_valid(type_schema, outer_evidence, accepted, rejected)
                continue

        if localizer_role == "SBV":
            targets = _direct_entities(children, entities_by_head, predicate_id, {"VOB"})
            target_role = "VOB"
        elif localizer_role in {"VOB", "POB"}:
            targets = _direct_entities(children, entities_by_head, predicate_id, {"SBV"})
            target_role = "SBV"
        else:
            targets = []
            target_role = None

        if len(anchors) != 1 or len(targets) != 1:
            rejected.append(_rejection(
                lexeme, trigger_id, "localizer", "incomplete_or_ambiguous_localizer_arguments",
                {"anchors": [item["entity_id"] for item in anchors],
                 "targets": [item["entity_id"] for item in targets],
                 "localizer_role": localizer_role},
            ))
            continue

        path = [
            {"dependent": anchors[0]["head_token_id"], "head": trigger_id, "relation": "ATT"},
            {"dependent": trigger_id, "head": heads.get(trigger_id), "relation": localizer_role},
            {"dependent": targets[0]["head_token_id"], "head": predicate_id, "relation": target_role},
        ]
        if preposition_id is not None:
            path.append({"preposition_token_id": preposition_id})
        evidence = _evidence_record(
            relation, lexeme, trigger_id, anchors[0], targets[0],
            "localizer_att_predicate", path,
        )
        _accept_if_valid(type_schema, evidence, accepted, rejected)

    return accepted, rejected


def extract_relation_evidence(row, type_schema):
    heads, dependency_relations, children = _dependency_maps(row)
    entities = build_entities(row, heads)
    accepted_direct, rejected_direct = _extract_direct_predicates(
        row, type_schema, entities, heads, dependency_relations, children
    )
    accepted_localizers, rejected_localizers = _extract_localizers(
        row, type_schema, entities, heads, dependency_relations, children
    )
    accepted_nested, rejected_nested = _extract_nested_overlap(
        row, type_schema, entities, heads, dependency_relations, children
    )

    accepted = []
    seen = set()
    for item in accepted_direct + accepted_localizers + accepted_nested:
        key = (
            item["relation"], _operand_key(item["left"]), _operand_key(item["right"])
        )
        if key not in seen:
            seen.add(key)
            accepted.append(item)

    unresolved_entities = [
        entity for entity in entities if entity.get("head_token_id") is None
    ]
    return {
        "sid": str(row.get("sid")),
        "sentence": str(row.get("sentence", "")),
        "entities": entities,
        "accepted": accepted,
        "rejected": rejected_direct + rejected_localizers + rejected_nested,
        "unresolved_entities": unresolved_entities,
    }
