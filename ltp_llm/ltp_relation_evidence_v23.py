#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V23 bottom-up LTP evidence with conservative composition support.

V23 inherits every accepted V22 candidate.  Its first new construction is
``whole 由 component(s) 组成/构成``.  The rule never falls back to a token
window or an implicit IN: both the construction and every operand must be
aligned to LTP dependencies and dataset entities.
"""

from __future__ import print_function

try:
    from . import ltp_relation_evidence_v22 as v22
except (ImportError, ValueError):
    import ltp_relation_evidence_v22 as v22


COMPOSITION_PREDICATES = {"组成", "构成"}
COMPOSITION_PREPOSITION = "由"
COMPOSITION_LINK_RELATIONS = {"ADV", "CMP", "FOB"}
COMPONENT_PATH_RELATIONS = {"POB", "COO", "VOB", "ATT"}
WHOLE_PATH_RELATIONS = {"SBV", "FOB", "ATT", "ADV", "VOB"}
INNER_OVERLAP_LEXEMES = {"夹", "夹有"}
TRAILING_SCOPE_RELATIONS = {
    lexeme: relation
    for lexeme, relation in v22.LOCALIZER_RELATIONS.items()
    if relation in {"IN", "A", "BW", "C"}
}


load_type_schema = v22.load_type_schema
schema_allows = v22.schema_allows


def _clause_ids(tokens):
    result = []
    clause_id = 0
    for token in tokens:
        result.append(clause_id)
        if token in v22.CLAUSE_PUNCTUATION:
            clause_id += 1
    return result


def _operand_depth(operand):
    if operand.get("kind") != "relation":
        return 0
    return 1 + max(
        _operand_depth(operand["left"]),
        _operand_depth(operand["right"]),
    )


def _operand_token_ids(operand):
    if operand.get("kind") != "relation":
        token_span = operand.get("token_span")
        if not token_span:
            return []
        return list(range(int(token_span[0]) + 1, int(token_span[1]) + 2))
    values = _operand_token_ids(operand["left"]) + _operand_token_ids(
        operand["right"]
    )
    trigger = operand.get("trigger_token_id")
    if trigger is not None:
        values.append(int(trigger))
    return sorted(set(values))


def _node_metadata(evidence, clause_ids):
    token_ids = _operand_token_ids(evidence["left"]) + _operand_token_ids(
        evidence["right"]
    )
    trigger_id = evidence.get("trigger_token_id")
    if trigger_id is not None:
        token_ids.append(int(trigger_id))
    token_ids = sorted(set(token_ids))
    clause_id = None
    if trigger_id is not None and 1 <= int(trigger_id) <= len(clause_ids):
        clause_id = clause_ids[int(trigger_id) - 1]
    return {
        "depth": 1 + max(
            _operand_depth(evidence["left"]),
            _operand_depth(evidence["right"]),
        ),
        "clause_id": clause_id,
        "token_scope": (
            [token_ids[0], token_ids[-1]] if token_ids else None
        ),
        "source": "ltp_dependency",
    }


def _annotate_evidence(evidence, clause_ids):
    result = dict(evidence)
    result["node"] = _node_metadata(evidence, clause_ids)
    return result


def _token_clause_bounds(tokens, token_id):
    index = token_id - 1
    left = index
    while left > 0 and tokens[left - 1] not in v22.CLAUSE_PUNCTUATION:
        left -= 1
    right = index
    while right + 1 < len(tokens) and tokens[right + 1] not in v22.CLAUSE_PUNCTUATION:
        right += 1
    return left + 1, right + 1


def _entity_before_token(entity, token_id):
    span = entity.get("token_span")
    return bool(span) and int(span[1]) + 1 < token_id


def _entity_between_tokens(entity, left_token_id, right_token_id):
    span = entity.get("token_span")
    if not span:
        return False
    start_id = int(span[0]) + 1
    end_id = int(span[1]) + 1
    return left_token_id < start_id and end_id < right_token_id


def _path_to_anchor(start_id, anchor_ids, heads, dependency_relations,
                    allowed_relations, max_steps=8):
    current = start_id
    path = []
    seen = set()
    for unused_step in range(max_steps):
        if current in anchor_ids:
            return path
        if current in seen or current <= 0:
            break
        seen.add(current)
        relation = dependency_relations.get(current)
        head = heads.get(current, 0)
        if relation not in allowed_relations or head <= 0:
            break
        path.append({
            "dependent": current,
            "head": head,
            "relation": relation,
        })
        current = head
    if current in anchor_ids:
        return path
    return None


def _unique_by_entity_id(entities):
    result = []
    seen = set()
    for entity in entities:
        entity_id = entity.get("entity_id")
        if entity_id not in seen:
            seen.add(entity_id)
            result.append(entity)
    return result


def _relation_node(relation, lexeme, trigger_id, left, right, pattern, path):
    return v22._relation_operand(
        relation, lexeme, trigger_id, left, right, pattern, path
    )


def _evidence(relation, lexeme, trigger_id, left, right, pattern, path):
    return v22._evidence_record(
        relation, lexeme, trigger_id, left, right, pattern, path
    )


def _accept(type_schema, evidence, accepted, rejected):
    v22._accept_if_valid(type_schema, evidence, accepted, rejected)


def _find_you_token(tokens, heads, dependency_relations, predicate_id,
                    clause_left):
    candidates = []
    for token_id in range(clause_left, predicate_id):
        if tokens[token_id - 1] != COMPOSITION_PREPOSITION:
            continue
        path = _path_to_anchor(
            token_id,
            {predicate_id},
            heads,
            dependency_relations,
            COMPOSITION_LINK_RELATIONS,
            max_steps=3,
        )
        if path is not None:
            candidates.append((token_id, path))
    if len(candidates) == 1:
        return candidates[0]
    return None, None


def _component_entities(entities, you_id, predicate_id, heads,
                        dependency_relations):
    result = []
    paths = {}
    for entity in entities:
        if not _entity_between_tokens(entity, you_id, predicate_id):
            continue
        head_id = entity.get("head_token_id")
        if head_id is None:
            continue
        path = _path_to_anchor(
            int(head_id),
            {you_id, predicate_id},
            heads,
            dependency_relations,
            COMPONENT_PATH_RELATIONS,
            max_steps=8,
        )
        if path is None:
            continue
        result.append(entity)
        paths[entity["entity_id"]] = path
    result.sort(key=lambda item: (
        item["token_span"][0], item["token_span"][1], item["entity_index"]
    ))
    return _unique_by_entity_id(result), paths


def _whole_entities_same_clause(entities, clause_left, you_id, predicate_id,
                                heads, dependency_relations):
    result = []
    paths = {}
    for entity in entities:
        span = entity.get("token_span")
        if not span:
            continue
        start_id = int(span[0]) + 1
        if start_id < clause_left or not _entity_before_token(entity, you_id):
            continue
        head_id = entity.get("head_token_id")
        if head_id is None:
            continue
        path = _path_to_anchor(
            int(head_id),
            {predicate_id},
            heads,
            dependency_relations,
            WHOLE_PATH_RELATIONS,
            max_steps=10,
        )
        if path is None:
            continue
        result.append(entity)
        paths[entity["entity_id"]] = path
    result.sort(key=lambda item: (
        item["token_span"][0], item["token_span"][1], item["entity_index"]
    ))
    return _unique_by_entity_id(result), paths


def _explicit_localizer_between(tokens, left_entity, right_entity, heads,
                                dependency_relations):
    left_end = int(left_entity["token_span"][1]) + 1
    right_start = int(right_entity["token_span"][0]) + 1
    candidates = []
    for token_id in range(left_end + 1, right_start):
        lexeme = tokens[token_id - 1]
        relation = TRAILING_SCOPE_RELATIONS.get(lexeme)
        if relation is None:
            continue
        left_path = _path_to_anchor(
            int(left_entity["head_token_id"]),
            {token_id}, heads, dependency_relations, {"ATT"}, max_steps=5,
        )
        right_path = _path_to_anchor(
            token_id,
            {int(right_entity["head_token_id"])},
            heads,
            dependency_relations,
            {"ATT"},
            max_steps=5,
        )
        if left_path is not None and right_path is not None:
            candidates.append((token_id, lexeme, relation, left_path + right_path))
    if len(candidates) == 1:
        return candidates[0]

    intervening_ids = list(range(left_end + 1, right_start))
    if len(intervening_ids) >= 2:
        compound = "".join(tokens[token_id - 1] for token_id in intervening_ids)
        relation = TRAILING_SCOPE_RELATIONS.get(compound)
        if relation is not None:
            right_head = int(right_entity["head_token_id"])
            left_path = _path_to_anchor(
                int(left_entity["head_token_id"]),
                {right_head}, heads, dependency_relations, {"ATT"}, max_steps=5,
            )
            token_paths = []
            for token_id in intervening_ids:
                token_path = _path_to_anchor(
                    token_id,
                    {right_head}, heads, dependency_relations,
                    {"ATT"}, max_steps=5,
                )
                if token_path is None:
                    token_paths = None
                    break
                token_paths.extend(token_path)
            if left_path is not None and token_paths is not None:
                return (
                    intervening_ids[0], compound, relation,
                    left_path + token_paths,
                )
    return None


def _build_whole_scope(tokens, whole_entities, heads, dependency_relations,
                       rejected, predicate_id):
    if not whole_entities:
        return None
    current = whole_entities[0]
    previous_entity = whole_entities[0]
    for next_entity in whole_entities[1:]:
        localizer = _explicit_localizer_between(
            tokens, previous_entity,
            next_entity, heads, dependency_relations
        )
        if localizer is None:
            # Only the last dependency-connected entity is the grammatical
            # whole when earlier entities cannot be joined by explicit scope.
            current = next_entity
            previous_entity = next_entity
            continue
        token_id, lexeme, relation, path = localizer
        current = _relation_node(
            relation, lexeme, token_id, current, next_entity,
            "composition_whole_scope", path,
        )
        previous_entity = next_entity
    return current


def _trailing_localizer(tokens, whole_entity, you_id, heads,
                        dependency_relations):
    start = int(whole_entity["token_span"][1]) + 2
    candidates = []
    for token_id in range(start, you_id):
        lexeme = tokens[token_id - 1]
        relation = TRAILING_SCOPE_RELATIONS.get(lexeme)
        if relation is None:
            continue
        path = _path_to_anchor(
            int(whole_entity["head_token_id"]),
            {token_id}, heads, dependency_relations, {"ATT"}, max_steps=5,
        )
        if path is None:
            continue
        if dependency_relations.get(token_id) not in {"ADV", "SBV", "FOB", "ATT"}:
            continue
        candidates.append((token_id, lexeme, relation, path + [{
            "dependent": token_id,
            "head": heads.get(token_id, 0),
            "relation": dependency_relations.get(token_id),
        }]))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _inner_overlap(tokens, components, you_id, predicate_id, heads,
                   dependency_relations):
    overlap_ids = [
        token_id for token_id in range(you_id + 1, predicate_id)
        if tokens[token_id - 1] in INNER_OVERLAP_LEXEMES
    ]
    if not overlap_ids:
        return None
    if len(overlap_ids) != 1:
        return "ambiguous"
    overlap_id = overlap_ids[0]
    left_values = [
        entity for entity in components
        if int(entity["token_span"][1]) + 1 < overlap_id
    ]
    right_values = [
        entity for entity in components
        if int(entity["token_span"][0]) + 1 > overlap_id
    ]
    if not left_values or not right_values:
        return "ambiguous"
    left = left_values[-1]
    right = right_values[0]
    right_path = _path_to_anchor(
        int(right["head_token_id"]),
        {overlap_id}, heads, dependency_relations,
        {"VOB", "POB", "ATT"}, max_steps=5,
    )
    left_path = _path_to_anchor(
        int(left["head_token_id"]),
        {you_id, overlap_id}, heads, dependency_relations,
        {"POB", "COO", "ATT"}, max_steps=5,
    )
    if left_path is None or right_path is None:
        return "ambiguous"
    return _relation_node(
        "PO", tokens[overlap_id - 1], overlap_id, left, right,
        "composition_inner_overlap", left_path + right_path,
    )


def _extract_compositions(row, type_schema, entities, heads,
                          dependency_relations, children):
    del children  # The validated head maps are sufficient for this rule.
    tokens = list(row.get("tokens", []))
    accepted = []
    rejected = []

    for predicate_id, predicate in enumerate(tokens, 1):
        if predicate not in COMPOSITION_PREDICATES:
            continue
        clause_left, unused_clause_right = _token_clause_bounds(tokens, predicate_id)
        you_id, you_path = _find_you_token(
            tokens, heads, dependency_relations, predicate_id, clause_left
        )
        if you_id is None:
            rejected.append(v22._rejection(
                predicate, predicate_id, "composition",
                "missing_unique_dependency_linked_由",
            ))
            continue

        components, component_paths = _component_entities(
            entities, you_id, predicate_id, heads, dependency_relations
        )
        if not components:
            rejected.append(v22._rejection(
                predicate, predicate_id, "composition",
                "missing_dependency_aligned_components",
            ))
            continue

        whole_entities, whole_paths = _whole_entities_same_clause(
            entities, clause_left, you_id, predicate_id,
            heads, dependency_relations,
        )
        if not whole_entities:
            rejected.append(v22._rejection(
                predicate, predicate_id, "composition",
                "missing_same_clause_dependency_aligned_whole",
                {"components": [item["entity_id"] for item in components]},
            ))
            continue

        whole = _build_whole_scope(
            tokens, whole_entities, heads, dependency_relations,
            rejected, predicate_id,
        )
        direct_whole = whole_entities[-1]
        trailing = _trailing_localizer(
            tokens, direct_whole, you_id, heads, dependency_relations
        )
        inner = _inner_overlap(
            tokens, components, you_id, predicate_id,
            heads, dependency_relations,
        )
        if inner == "ambiguous":
            rejected.append(v22._rejection(
                predicate, predicate_id, "composition",
                "ambiguous_inner_overlap",
            ))
            continue

        component_operands = [inner] if inner is not None else components
        if inner is not None:
            inner_evidence = _evidence(
                "PO", inner["lexeme"], inner["trigger_token_id"],
                inner["left"], inner["right"],
                inner["pattern"], inner["dependency_path"],
            )
            _accept(type_schema, inner_evidence, accepted, rejected)

        for component in component_operands:
            relation = "IN"
            lexeme = predicate
            trigger_id = predicate_id
            pattern = "composition_whole_first"
            left = whole
            path = list(you_path or [])
            path.extend(whole_paths.get(direct_whole["entity_id"], []))
            if component.get("kind") == "relation":
                path.extend(component.get("dependency_path", []))
            else:
                path.extend(component_paths.get(component["entity_id"], []))

            if trailing is not None:
                trigger_id, lexeme, relation, trailing_path = trailing
                pattern = "composition_trailing_scope"
                path.extend(trailing_path)
            evidence = _evidence(
                relation, lexeme, trigger_id, left, component, pattern, path
            )
            _accept(type_schema, evidence, accepted, rejected)

    return accepted, rejected


def extract_relation_evidence(row, type_schema, enable_composition=True):
    base = v22.extract_relation_evidence(row, type_schema)
    tokens = list(row.get("tokens", []))
    clause_ids = _clause_ids(tokens)

    if not enable_composition:
        result = dict(base)
        result["accepted"] = [
            _annotate_evidence(item, clause_ids)
            for item in base["accepted"]
        ]
        result["version"] = "v23_compatibility"
        return result

    heads, dependency_relations, children = v22._dependency_maps(row)
    entities = base["entities"]
    accepted_composition, rejected_composition = _extract_compositions(
        row, type_schema, entities, heads, dependency_relations, children
    )

    accepted = []
    seen = set()
    for item in list(base["accepted"]) + accepted_composition:
        key = (
            item["relation"],
            v22._operand_key(item["left"]),
            v22._operand_key(item["right"]),
        )
        if key in seen:
            continue
        seen.add(key)
        accepted.append(_annotate_evidence(item, clause_ids))

    # Replace the generic V22 composition deferral with V23's factual reason.
    rejected = [
        item for item in base["rejected"]
        if item.get("reason") != "deferred_composition_rule"
    ]
    rejected.extend(rejected_composition)
    result = dict(base)
    result["accepted"] = accepted
    result["rejected"] = rejected
    result["version"] = "v23"
    return result
