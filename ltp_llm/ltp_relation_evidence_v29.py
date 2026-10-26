#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V29 controlled expansion over the frozen V28 candidate set.

The expansion relaxes uniqueness checks inside two already enabled explicit
families.  It does not enable COO/ATT propagation, implicit IN, cross-clause
completion, default relations, or token-window pairing.  Newly recovered L1
relations are passed through the same V28 bottom-up closure builders.
"""

from __future__ import print_function

try:
    from . import ltp_relation_evidence_v22 as v22
    from . import ltp_relation_evidence_v23 as v23
    from . import ltp_relation_evidence_v24 as v24
    from . import ltp_relation_evidence_v25 as v25
    from . import ltp_relation_evidence_v27 as v27
    from . import ltp_relation_evidence_v28 as v28
except (ImportError, ValueError):
    import ltp_relation_evidence_v22 as v22
    import ltp_relation_evidence_v23 as v23
    import ltp_relation_evidence_v24 as v24
    import ltp_relation_evidence_v25 as v25
    import ltp_relation_evidence_v27 as v27
    import ltp_relation_evidence_v28 as v28


load_type_schema = v28.load_type_schema
schema_allows = v28.schema_allows

EXPANDED_LOCATIVE_PREDICATES = set(v27.LOCATIVE_PREDICATES) | {
    "位于", "处于", "分布于", "出露于", "见于",
}


def _entity_start(entity):
    return int(entity["token_span"][0]) + 1


def _entity_end(entity):
    return int(entity["token_span"][1]) + 1


def _dependency_path(left, right, heads, dependency_relations, max_steps=10):
    return v24._dependency_path_between(
        int(left["head_token_id"]), int(right["head_token_id"]),
        heads, dependency_relations,
        v25.EXPLICIT_PATH_RELATIONS | {"FOB", "RAD", "DBL"},
        max_steps=max_steps,
    )


def _accept(type_schema, relation, lexeme, trigger_id, left, right,
            pattern, path, accepted, rejected):
    evidence = v22._evidence_record(
        relation, lexeme, trigger_id, left, right, pattern, path
    )
    v22._accept_if_valid(type_schema, evidence, accepted, rejected)


def _inside_parentheses(tokens, token_id):
    depth = 0
    for index, token in enumerate(tokens, 1):
        if index > token_id:
            break
        if token in {"（", "(", "【", "["}:
            depth += 1
        elif token in {"）", ")", "】", "]"} and depth:
            depth -= 1
    return depth > 0


def _extract_unique_locative_directions(row, base, type_schema):
    """Recover the dataset's anchor-to-local-object direction convention."""
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    entities = sorted(
        base["entities"], key=lambda item: (_entity_start(item), _entity_end(item))
    )
    accepted = []
    rejected = []
    for trigger_id, lexeme in enumerate(tokens, 1):
        relation = v27.DIRECTION_LOCALIZERS.get(lexeme)
        if relation is None or _inside_parentheses(tokens, trigger_id):
            continue
        previous = tokens[trigger_id - 2] if trigger_id > 1 else ""
        following = tokens[trigger_id] if trigger_id < len(tokens) else ""
        if previous in v27.DIRECTION_LOCALIZERS or following in v27.DIRECTION_LOCALIZERS:
            continue
        clause = clause_ids[trigger_id - 1]
        predicates = [
            token_id for token_id in range(1, trigger_id)
            if clause_ids[token_id - 1] == clause
            and tokens[token_id - 1] in EXPANDED_LOCATIVE_PREDICATES
        ]
        if not predicates:
            continue
        predicate_id = predicates[-1]
        direction_count = sum(
            1 for token_id in range(predicate_id + 1, len(tokens) + 1)
            if clause_ids[token_id - 1] == clause
            and tokens[token_id - 1] in v27.DIRECTION_LOCALIZERS
        )
        if direction_count != 1:
            continue
        same_clause = [
            entity for entity in entities
            if entity.get("head_token_id") is not None
            and clause_ids[int(entity["head_token_id"]) - 1] == clause
        ]
        subjects = [
            entity for entity in same_clause
            if _entity_end(entity) < predicate_id
        ]
        references = [
            entity for entity in same_clause
            if predicate_id < _entity_start(entity) < trigger_id
        ]
        if len(subjects) != 1 or len(references) != 1:
            continue
        reference = references[0]
        following_entities = sorted(
            [
                entity for entity in same_clause
                if _entity_start(entity) > trigger_id
            ],
            key=lambda item: (_entity_start(item), _entity_end(item)),
        )
        if not following_entities:
            continue
        left = reference
        right = following_entities[0]
        pattern = "expanded_embedded_locative_direction"
        sentence = row.get("sentence", "")
        between = sentence[
            int(reference["span"][1]) + 1:int(right["span"][0])
        ]
        for variants, diagonal in (
            (("东南", "南东"), "SE"),
            (("东北", "北东"), "NE"),
            (("西南", "南西"), "SW"),
            (("西北", "北西"), "NW"),
        ):
            if any(value in between for value in variants):
                relation = diagonal
                break
        path = _dependency_path(
            left, right, heads, dependency_relations, max_steps=10
        )
        if path is None:
            continue
        _accept(
            type_schema, relation, lexeme, trigger_id, left, right,
            pattern, path,
            accepted, rejected,
        )
    return accepted, rejected


def _merge(items, clause_ids):
    accepted = []
    seen = set()
    for item in items:
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
    return accepted


def extract_relation_evidence(row, type_schema, enable_expansion=True):
    base = v28.extract_relation_evidence(row, type_schema, True)
    if not enable_expansion:
        result = dict(base)
        result["version"] = "v29_compatibility"
        return result

    additions = []
    rejections = []
    for extractor in (_extract_unique_locative_directions,):
        added, rejected = extractor(row, base, type_schema)
        additions.extend(added)
        rejections.extend(rejected)

    clause_ids = v23._clause_ids(list(row.get("tokens", [])))
    result = dict(base)
    # V28 L2/L3 remains frozen.  The lower-confidence expansion is not used
    # as a new closure seed until it has independently passed L1 evaluation.
    result["accepted"] = _merge(list(base["accepted"]) + additions, clause_ids)
    result["rejected"] = list(base["rejected"]) + rejections
    result["version"] = "v29_expanded"
    return result
