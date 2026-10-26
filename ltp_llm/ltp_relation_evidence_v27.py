#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V27 explicit L1 completion over the frozen V26 candidate set.

V27 adds no new parent closure.  It only accepts surface-explicit L1
relations from four reusable families: clause-local copular identity,
parenthetical aliases, stratigraphic A/BW/C links and directional localizers.
Every candidate still needs a schema-compatible entity pair and an LTP
dependency path; there is no default IN, token window fallback or SID rule.
"""

from __future__ import print_function

try:
    from . import ltp_relation_evidence_v22 as v22
    from . import ltp_relation_evidence_v23 as v23
    from . import ltp_relation_evidence_v24 as v24
    from . import ltp_relation_evidence_v25 as v25
    from . import ltp_relation_evidence_v26 as v26
except (ImportError, ValueError):
    import ltp_relation_evidence_v22 as v22
    import ltp_relation_evidence_v23 as v23
    import ltp_relation_evidence_v24 as v24
    import ltp_relation_evidence_v25 as v25
    import ltp_relation_evidence_v26 as v26


load_type_schema = v26.load_type_schema
schema_allows = v26.schema_allows

COPULAR_LEXEMES = {"为", "是", "属于", "属"}
POSITIONAL_RELATIONS = {"A", "BW", "C"}
DIRECTION_RELATIONS = {"N", "S", "E", "W", "NE", "SE", "NW", "SW"}
DIRECTION_LOCALIZERS = {
    lexeme: relation
    for relation in DIRECTION_RELATIONS
    for lexeme in v22.RELATION_LEXICON[relation]
}
DIRECTION_LOCALIZERS.update({
    "北东": "NE", "北东部": "NE", "北东侧": "NE", "北东翼": "NE",
    "南东": "SE", "南东部": "SE", "南东侧": "SE", "南东翼": "SE",
    "北西": "NW", "北西部": "NW", "北西侧": "NW", "北西翼": "NW",
    "南西": "SW", "南西部": "SW", "南西侧": "SW", "南西翼": "SW",
    "东北角": "NE", "东南角": "SE", "西北角": "NW", "西南角": "SW",
    "北翼": "N", "南翼": "S", "东翼": "E", "西翼": "W",
    "中南部": "S",
})
LOCATIVE_PREDICATES = {"位于", "分布", "出露", "展布", "处于"}


def _entity_start(entity):
    return int(entity["token_span"][0]) + 1


def _entity_end(entity):
    return int(entity["token_span"][1]) + 1


def _same_clause_entities(entities, token_id, clause_ids):
    clause = clause_ids[token_id - 1]
    return [
        entity for entity in entities
        if entity.get("head_token_id") is not None
        and clause_ids[int(entity["head_token_id"]) - 1] == clause
    ]


def _dependency_path(left, right, heads, dependency_relations, max_steps=10):
    return v24._dependency_path_between(
        int(left["head_token_id"]), int(right["head_token_id"]),
        heads, dependency_relations,
        v25.EXPLICIT_PATH_RELATIONS | {"FOB", "RAD", "DBL"},
        max_steps=max_steps,
    )


def _accept_entity_relation(type_schema, relation, lexeme, trigger_id,
                            left, right, pattern, path, accepted, rejected):
    evidence = v22._evidence_record(
        relation, lexeme, trigger_id, left, right, pattern, path
    )
    v22._accept_if_valid(type_schema, evidence, accepted, rejected)


def _extract_clause_copular(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []
    entities = base["entities"]

    for trigger_id, lexeme in enumerate(tokens, 1):
        if lexeme not in COPULAR_LEXEMES:
            continue
        clause_entities = _same_clause_entities(entities, trigger_id, clause_ids)
        before = [
            entity for entity in clause_entities
            if _entity_end(entity) < trigger_id
        ]
        after = [
            entity for entity in clause_entities
            if _entity_start(entity) > trigger_id
        ]
        if len(before) != 1 or len(after) != 1:
            continue
        left, right = before[0], after[0]
        left_head = int(left["head_token_id"])
        right_head = int(right["head_token_id"])
        if not (
            heads.get(left_head) == trigger_id
            and dependency_relations.get(left_head) == "SBV"
            and heads.get(right_head) == trigger_id
            and dependency_relations.get(right_head) in {"VOB", "POB"}
        ):
            continue
        path = _dependency_path(
            left, right, heads, dependency_relations, max_steps=8
        )
        if path is None:
            continue
        _accept_entity_relation(
            type_schema, "EQ", lexeme, trigger_id, left, right,
            "explicit_clause_copular_identity", path, accepted, rejected,
        )
    return accepted, rejected


def _extract_parenthetical_aliases(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    accepted = []
    rejected = []
    ordered = sorted(
        base["entities"], key=lambda item: (_entity_start(item), _entity_end(item))
    )
    for left, right in zip(ordered, ordered[1:]):
        if left.get("type") != "RK" or right.get("type") != "RK":
            continue
        between = tokens[_entity_end(left):_entity_start(right) - 1]
        after_id = _entity_end(right) + 1
        if between not in (["（"], ["("], ["【"], ["["]):
            continue
        if after_id > len(tokens) or tokens[after_id - 1] not in {"）", ")", "】", "]"}:
            continue
        path = _dependency_path(
            left, right, heads, dependency_relations, max_steps=4
        )
        if path is None:
            continue
        _accept_entity_relation(
            type_schema, "EQ", tokens[_entity_end(left)],
            _entity_end(left) + 1, left, right,
            "explicit_parenthetical_alias", path, accepted, rejected,
        )
    return accepted, rejected


def _extract_positional_entity_links(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []
    ordered = sorted(
        base["entities"], key=lambda item: (_entity_start(item), _entity_end(item))
    )
    for pair_index, (left, right) in enumerate(zip(ordered, ordered[1:])):
        link = v26._pair_hierarchy_link(
            tokens, left, right, heads, dependency_relations, clause_ids
        )
        if link is None or link[0] not in POSITIONAL_RELATIONS:
            continue
        # Do not emit a truncated positional edge when the right entity is an
        # operand of an already accepted relation.  V28 must wrap that whole
        # relation instead (BW(X, PO(A, B)), not BW(X, A)).
        right_id = right.get("entity_id")
        if any(
            item.get("relation") == "PO"
            and right_id in v24._operand_entity_ids(v26._relation_operand(item))
            for item in base["accepted"]
            if int((item.get("node") or {}).get("depth", 0)) == 1
        ):
            continue
        # Likewise, a preceding explicit scope makes ``left`` an incomplete
        # operand.  The complete chain is constructed only in V28.
        if pair_index > 0:
            prefix = v26._pair_hierarchy_link(
                tokens, ordered[pair_index - 1], left,
                heads, dependency_relations, clause_ids,
            )
            if prefix is not None:
                continue
        relation, lexeme, trigger_id, unused_pattern, path = link
        if lexeme == "之上" and right.get("type") == "RK":
            continue
        _accept_entity_relation(
            type_schema, relation, lexeme, trigger_id, left, right,
            "explicit_standalone_positional_link", path, accepted, rejected,
        )

    for trigger_id, lexeme in enumerate(tokens, 1):
        relation = v26.V26_POSITIONAL_LOCALIZERS.get(lexeme)
        if relation not in {"A", "BW"}:
            continue
        clause = clause_ids[trigger_id - 1]
        values = [
            entity for entity in ordered
            if entity.get("head_token_id") is not None
            and clause_ids[int(entity["head_token_id"]) - 1] == clause
            and _entity_end(entity) < trigger_id
        ]
        if len(values) != 2:
            continue
        subject, anchor = values[0], values[1]
        between = tokens[_entity_end(subject):_entity_start(anchor) - 1]
        if "于" not in between:
            continue
        path = _dependency_path(
            anchor, subject, heads, dependency_relations, max_steps=8
        )
        if path is None:
            continue
        _accept_entity_relation(
            type_schema, relation, lexeme, trigger_id, anchor, subject,
            "explicit_postposed_positional_link", path, accepted, rejected,
        )
    return accepted, rejected


def _extract_directional_links(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    entities = sorted(
        base["entities"], key=lambda item: (_entity_start(item), _entity_end(item))
    )
    accepted = []
    rejected = []

    def inside_parentheses(token_id):
        depth = 0
        for index, token in enumerate(tokens, 1):
            if index > token_id:
                break
            if token in {"（", "(", "【", "["}:
                depth += 1
            elif token in {"）", ")", "】", "]"} and depth:
                depth -= 1
        return depth > 0

    for trigger_id, lexeme in enumerate(tokens, 1):
        relation = DIRECTION_LOCALIZERS.get(lexeme)
        if relation is None:
            continue
        if inside_parentheses(trigger_id):
            continue
        previous = tokens[trigger_id - 2] if trigger_id > 1 else ""
        following = tokens[trigger_id] if trigger_id < len(tokens) else ""
        if (
            previous in DIRECTION_LOCALIZERS
            or following in DIRECTION_LOCALIZERS
        ):
            continue
        clause = clause_ids[trigger_id - 1]
        same_clause = [
            entity for entity in entities
            if entity.get("head_token_id") is not None
            and clause_ids[int(entity["head_token_id"]) - 1] == clause
        ]
        before = [entity for entity in same_clause if _entity_end(entity) < trigger_id]
        after = [entity for entity in same_clause if _entity_start(entity) > trigger_id]
        left = None
        right = None
        pattern = None

        locative_ids = [
            token_id for token_id in range(1, trigger_id)
            if tokens[token_id - 1] in LOCATIVE_PREDICATES
            and clause_ids[token_id - 1] == clause
        ]
        if locative_ids:
            predicate_id = locative_ids[-1]
            subjects = [
                entity for entity in same_clause
                if _entity_end(entity) < predicate_id
            ]
            references = [
                entity for entity in same_clause
                if predicate_id < _entity_start(entity) < trigger_id
            ]
            directional_count = sum(
                1 for token_id in range(predicate_id + 1, len(tokens) + 1)
                if clause_ids[token_id - 1] == clause
                and tokens[token_id - 1] in DIRECTION_LOCALIZERS
            )
            keep_locative = (
                lexeme.endswith("角") or lexeme == "中南部"
                or directional_count >= 2
            )
            if keep_locative and len(subjects) == 1 and references:
                left = subjects[0]
                right = references[-1]
                pattern = "explicit_locative_direction"

        if left is None:
            copular_ids = [
                token_id for token_id in range(1, trigger_id)
                if tokens[token_id - 1] in COPULAR_LEXEMES
                and clause_ids[token_id - 1] == clause
            ]
            if copular_ids:
                copular_id = copular_ids[-1]
                subjects = [
                    entity for entity in same_clause
                    if _entity_end(entity) < copular_id
                ]
                anchors = [
                    entity for entity in same_clause
                    if copular_id < _entity_start(entity) < trigger_id
                ]
                if len(subjects) == 1 and len(anchors) == 1:
                    left = anchors[0]
                    right = subjects[0]
                    pattern = "explicit_copular_direction"

        if left is None and not locative_ids:
            if len(before) == 1 and len(after) == 1:
                left, right = before[0], after[0]
                pattern = "explicit_infix_direction"

        if left is None or right is None or left["entity_id"] == right["entity_id"]:
            continue
        path = _dependency_path(
            left, right, heads, dependency_relations, max_steps=10
        )
        if path is None:
            continue
        _accept_entity_relation(
            type_schema, relation, lexeme, trigger_id, left, right,
            pattern, path, accepted, rejected,
        )
    return accepted, rejected


def extract_relation_evidence(row, type_schema, enable_explicit_l1=True):
    base = v26.extract_relation_evidence(
        row, type_schema, enable_parent_closure=True
    )
    if not enable_explicit_l1:
        result = dict(base)
        result["version"] = "v27_compatibility"
        return result

    families = [
        _extract_clause_copular,
        _extract_parenthetical_aliases,
        _extract_positional_entity_links,
        _extract_directional_links,
    ]
    additions = []
    rejections = []
    for extractor in families:
        accepted, rejected = extractor(row, base, type_schema)
        additions.extend(accepted)
        rejections.extend(rejected)

    clause_ids = v23._clause_ids(list(row.get("tokens", [])))
    accepted = []
    seen = set()
    for item in list(base["accepted"]) + additions:
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
    result["rejected"] = list(base["rejected"]) + rejections
    result["version"] = "v27"
    return result
