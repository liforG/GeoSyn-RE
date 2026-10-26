#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V24 conservative wrappers around an existing V23 relation.

The rule is bottom-up: V23 must already have accepted the inner PO.  V24 only
adds A/BW/C when an explicit localizer has one ATT-linked anchor entity and
the localizer plus PO predicate share one dependency spine in the same clause.
It can also add IN around one already accepted L1 relation when an explicit
carrier construction (``X见Y``, ``X中发育Y`` or ``X地区发育Y``) closes the
dependency path between X, the carrier predicate and Y.
No cross-clause carry, window lookup, default IN, or layer enumeration is used.
"""

from __future__ import print_function

try:
    from . import ltp_relation_evidence_v22 as v22
    from . import ltp_relation_evidence_v23 as v23
except (ImportError, ValueError):
    import ltp_relation_evidence_v22 as v22
    import ltp_relation_evidence_v23 as v23


POSITIONAL_LOCALIZERS = {
    lexeme: relation
    for lexeme, relation in v22.LOCALIZER_RELATIONS.items()
    if relation in {"A", "BW", "C"}
}
OUTER_SCOPE_LEXEMES = {"内", "附近"}
LOCALIZER_TO_SPINE_RELATIONS = {"ATT", "SBV", "ADV"}
CARRIER_PREDICATES = {"发育", "见", "可见"}
CARRIER_LOCALIZERS = {"中", "内", "之中", "内部"}
CARRIER_INNER_RELATIONS = {"PO", "A", "BW", "C"}
CARRIER_LINK_RELATIONS = {"ATT", "SBV", "VOB", "COO", "ADV"}
REGION_NOUNS = {"地区", "区"}

load_type_schema = v23.load_type_schema
schema_allows = v23.schema_allows


def _relation_operand(item):
    return v22._relation_operand(
        item["relation"], item["lexeme"], item["trigger_token_id"],
        item["left"], item["right"], item["pattern"],
        item.get("dependency_path", []),
    )


def _anchor_entities(entities, localizer_id, clause_ids, heads,
                     dependency_relations):
    result = []
    paths = {}
    localizer_clause = clause_ids[localizer_id - 1]
    for entity in entities:
        head_id = entity.get("head_token_id")
        if head_id is None or clause_ids[int(head_id) - 1] != localizer_clause:
            continue
        path = v23._path_to_anchor(
            int(head_id), {localizer_id}, heads, dependency_relations,
            {"ATT"}, max_steps=4,
        )
        if path is None or not path:
            continue
        result.append(entity)
        paths[entity["entity_id"]] = path
    result = v23._unique_by_entity_id(result)
    if not result:
        return result, paths
    shortest = min(len(paths[item["entity_id"]]) for item in result)
    nearest = [
        item for item in result
        if len(paths[item["entity_id"]]) == shortest
    ]
    return nearest, paths


def _po_spine_anchor(po_operand, heads):
    trigger_id = int(po_operand["trigger_token_id"])
    head_id = int(heads.get(trigger_id, 0))
    return head_id if head_id > 0 else trigger_id


def _matching_po_operands(base, localizer_id, clause_ids, heads,
                          dependency_relations):
    result = []
    paths = {}
    localizer_clause = clause_ids[localizer_id - 1]
    for item in base["accepted"]:
        if item.get("relation") != "PO":
            continue
        operand = _relation_operand(item)
        token_ids = v23._operand_token_ids(operand)
        if not token_ids or min(token_ids) <= localizer_id:
            continue
        if any(clause_ids[token_id - 1] != localizer_clause for token_id in token_ids):
            continue
        spine_anchor = _po_spine_anchor(operand, heads)
        path = v23._path_to_anchor(
            localizer_id, {spine_anchor}, heads, dependency_relations,
            LOCALIZER_TO_SPINE_RELATIONS, max_steps=4,
        )
        if path is None or not path:
            continue
        key = v22._operand_key(operand)
        result.append(operand)
        paths[key] = path
    unique = []
    seen = set()
    for operand in result:
        key = v22._operand_key(operand)
        if key not in seen:
            seen.add(key)
            unique.append(operand)
    return unique, paths


def _operand_entity_ids(operand):
    if operand.get("kind") != "relation":
        return {operand.get("entity_id")}
    return _operand_entity_ids(operand["left"]) | _operand_entity_ids(
        operand["right"]
    )


def _dependency_path_between(start_id, target_id, heads,
                             dependency_relations, allowed_relations,
                             max_steps=3):
    """Return one short undirected dependency path without token windows."""
    adjacency = {}
    for dependent, head in heads.items():
        relation = dependency_relations.get(dependent)
        if head <= 0 or relation not in allowed_relations:
            continue
        edge = {
            "dependent": dependent,
            "head": head,
            "relation": relation,
        }
        adjacency.setdefault(dependent, []).append((head, edge))
        adjacency.setdefault(head, []).append((dependent, edge))
    queue = [(int(start_id), [])]
    seen = {int(start_id)}
    while queue:
        current, path = queue.pop(0)
        if current == int(target_id):
            return path
        if len(path) >= max_steps:
            continue
        for next_id, edge in adjacency.get(current, []):
            if next_id in seen:
                continue
            seen.add(next_id)
            queue.append((next_id, path + [edge]))
    return None


def _carrier_inner_relations(base, predicate_id, clause_ids, heads,
                             dependency_relations):
    values = []
    paths = {}
    predicate_clause = clause_ids[predicate_id - 1]
    for item in base["accepted"]:
        if item.get("relation") not in CARRIER_INNER_RELATIONS:
            continue
        if int((item.get("node") or {}).get("depth", 0)) != 1:
            continue
        trigger_id = item.get("trigger_token_id")
        if trigger_id is None or clause_ids[int(trigger_id) - 1] != predicate_clause:
            continue
        path = _dependency_path_between(
            int(trigger_id), predicate_id, heads, dependency_relations,
            CARRIER_LINK_RELATIONS, max_steps=2,
        )
        if path is None or not path:
            continue
        operand = _relation_operand(item)
        key = v22._operand_key(operand)
        values.append(operand)
        paths[key] = path
    unique = []
    seen = set()
    for operand in values:
        key = v22._operand_key(operand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(operand)
    return unique, paths


def _direct_carrier_entities(entities, predicate_id, heads,
                             dependency_relations, clause_ids):
    predicate_clause = clause_ids[predicate_id - 1]
    return v23._unique_by_entity_id([
        entity for entity in entities
        if entity.get("head_token_id") is not None
        and clause_ids[int(entity["head_token_id"]) - 1] == predicate_clause
        and heads.get(int(entity["head_token_id"])) == predicate_id
        and dependency_relations.get(int(entity["head_token_id"])) == "SBV"
    ])


def _localizer_carrier_entities(entities, localizer_id, heads,
                                dependency_relations, clause_ids):
    values = []
    paths = {}
    localizer_clause = clause_ids[localizer_id - 1]
    for entity in entities:
        head_id = entity.get("head_token_id")
        if head_id is None or int(head_id) >= localizer_id:
            continue
        if clause_ids[int(head_id) - 1] != localizer_clause:
            continue
        path = v23._path_to_anchor(
            int(head_id), {localizer_id}, heads, dependency_relations,
            {"ATT"}, max_steps=5,
        )
        if path is None or not path:
            continue
        values.append(entity)
        paths[entity["entity_id"]] = path
    values = v23._unique_by_entity_id(values)
    if not values:
        return values, paths
    shortest = min(len(paths[item["entity_id"]]) for item in values)
    return [
        item for item in values
        if len(paths[item["entity_id"]]) == shortest
    ], paths


def _nominal_region_carriers(tokens, entities, predicate_id, inner,
                             heads, dependency_relations, clause_ids):
    """Recognise the dependency-closed ``X地区发育Y`` nominal structure."""
    if tokens[predicate_id - 1] != "发育":
        return [], {}
    if dependency_relations.get(predicate_id) != "ATT":
        return [], {}
    shared_head = heads.get(predicate_id, 0)
    if shared_head <= 0 or shared_head not in v23._operand_token_ids(inner):
        return [], {}
    values = []
    paths = {}
    for entity in entities:
        head_id = entity.get("head_token_id")
        if head_id is None or int(head_id) >= predicate_id:
            continue
        if clause_ids[int(head_id) - 1] != clause_ids[predicate_id - 1]:
            continue
        path = v23._path_to_anchor(
            int(head_id), {shared_head}, heads, dependency_relations,
            {"ATT"}, max_steps=3,
        )
        if path is None or len(path) != 2:
            continue
        intermediate_ids = [edge["head"] for edge in path[:-1]]
        if len(intermediate_ids) != 1:
            continue
        if tokens[intermediate_ids[0] - 1] not in REGION_NOUNS:
            continue
        values.append(entity)
        paths[entity["entity_id"]] = path + [{
            "dependent": predicate_id,
            "head": shared_head,
            "relation": "ATT",
        }]
    return v23._unique_by_entity_id(values), paths


def _carrier_has_pending_positional_scope(tokens, entities, carrier, heads,
                                          dependency_relations):
    """Do not flatten ``X顶部Y见...`` into ``IN(Y, ...)``.

    A/BW/C localizers make the complete left operand a positional relation.
    If V24 has not constructed that relation yet, retaining the bare carrier
    would skip a required node.  Generic IN localizers are intentionally not
    blocked because GNRED retains the inner carrier relation in ``X中的Y``.
    """
    carrier_head = int(carrier["head_token_id"])
    for localizer_id, lexeme in enumerate(tokens, 1):
        if POSITIONAL_LOCALIZERS.get(lexeme) not in {"A", "BW", "C"}:
            continue
        if heads.get(localizer_id) != carrier_head:
            continue
        if dependency_relations.get(localizer_id) != "ATT":
            continue
        for entity in entities:
            head_id = entity.get("head_token_id")
            if head_id is None or entity["entity_id"] == carrier["entity_id"]:
                continue
            path = v23._path_to_anchor(
                int(head_id), {localizer_id}, heads, dependency_relations,
                {"ATT"}, max_steps=4,
            )
            if path is not None and path:
                return True
    return False


def _extract_carrier_wrappers(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []

    for predicate_id, predicate in enumerate(tokens, 1):
        inner_values, inner_paths = _carrier_inner_relations(
            base, predicate_id, clause_ids, heads, dependency_relations
        )
        if len(inner_values) != 1:
            continue
        inner = inner_values[0]

        carrier_options = []
        if predicate in CARRIER_PREDICATES:
            direct = _direct_carrier_entities(
                base["entities"], predicate_id, heads,
                dependency_relations, clause_ids,
            )
            if len(direct) == 1:
                carrier_options.append((
                    direct[0], predicate, predicate_id,
                    "explicit_carrier_predicate", [{
                        "dependent": direct[0]["head_token_id"],
                        "head": predicate_id,
                        "relation": "SBV",
                    }],
                ))

        cross_clause_localizers = []
        for localizer_id, lexeme in enumerate(tokens, 1):
            if lexeme not in CARRIER_LOCALIZERS:
                continue
            if heads.get(localizer_id) != predicate_id:
                continue
            if dependency_relations.get(localizer_id) not in {"ADV", "SBV"}:
                continue
            anchors, anchor_paths = _localizer_carrier_entities(
                base["entities"], localizer_id, heads,
                dependency_relations, clause_ids,
            )
            if len(anchors) != 1:
                continue
            option = (
                anchors[0], lexeme, localizer_id,
                "explicit_localizer_carrier",
                anchor_paths[anchors[0]["entity_id"]] + [{
                    "dependent": localizer_id,
                    "head": predicate_id,
                    "relation": dependency_relations.get(localizer_id),
                }],
            )
            if clause_ids[localizer_id - 1] == clause_ids[predicate_id - 1]:
                carrier_options.append(option)
            else:
                cross_clause_localizers.append(option)

        nominal, nominal_paths = _nominal_region_carriers(
            tokens, base["entities"], predicate_id, inner,
            heads, dependency_relations, clause_ids,
        )
        if len(nominal) == 1:
            carrier_options.append((
                nominal[0], predicate, predicate_id,
                "explicit_region_carrier",
                nominal_paths[nominal[0]["entity_id"]],
            ))

        if not carrier_options and len(cross_clause_localizers) == 1:
            carrier_options.extend(cross_clause_localizers)

        unique_options = []
        seen_carriers = set()
        for option in carrier_options:
            carrier = option[0]
            if carrier["entity_id"] in _operand_entity_ids(inner):
                continue
            if _carrier_has_pending_positional_scope(
                tokens, base["entities"], carrier,
                heads, dependency_relations,
            ):
                rejected.append(v22._rejection(
                    option[1], option[2], "carrier_wrapper",
                    "carrier_has_unbuilt_positional_scope",
                    {"carrier": carrier["entity_id"]},
                ))
                continue
            if carrier["entity_id"] in seen_carriers:
                continue
            seen_carriers.add(carrier["entity_id"])
            unique_options.append(option)
        if len(unique_options) != 1:
            if unique_options:
                rejected.append(v22._rejection(
                    predicate, predicate_id, "carrier_wrapper",
                    "carrier_wrapper_arguments_not_unique",
                    {"carriers": [item[0]["entity_id"] for item in unique_options]},
                ))
            continue

        carrier, lexeme, trigger_id, pattern, carrier_path = unique_options[0]
        path = list(carrier_path)
        path.extend(inner_paths[v22._operand_key(inner)])
        path.extend(inner.get("dependency_path", []))
        evidence = v22._evidence_record(
            "IN", lexeme, trigger_id, carrier, inner, pattern, path
        )
        v22._accept_if_valid(type_schema, evidence, accepted, rejected)
    return accepted, rejected


def _has_intervening_entity(entities, anchor, po_operand, localizer_id):
    po_token_ids = v23._operand_token_ids(po_operand)
    if not po_token_ids:
        return True
    first_po_token = min(po_token_ids)
    allowed_ids = _operand_entity_ids(po_operand) | {anchor.get("entity_id")}
    for entity in entities:
        if entity.get("entity_id") in allowed_ids:
            continue
        span = entity.get("token_span")
        if not span:
            continue
        start_id = int(span[0]) + 1
        if localizer_id < start_id < first_po_token:
            return True
    return False


def _extract_positional_wrappers(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []

    for localizer_id, lexeme in enumerate(tokens, 1):
        relation = POSITIONAL_LOCALIZERS.get(lexeme)
        if relation is None:
            continue
        anchors, anchor_paths = _anchor_entities(
            base["entities"], localizer_id, clause_ids,
            heads, dependency_relations,
        )
        po_values, po_paths = _matching_po_operands(
            base, localizer_id, clause_ids, heads, dependency_relations
        )
        if len(anchors) != 1 or len(po_values) != 1:
            rejected.append(v22._rejection(
                lexeme, localizer_id, "positional_po_wrapper",
                "positional_wrapper_arguments_not_unique",
                {
                    "anchors": [item["entity_id"] for item in anchors],
                    "po_operands": [v22._operand_key(item) for item in po_values],
                },
            ))
            continue
        anchor = anchors[0]
        po_operand = po_values[0]
        if _has_intervening_entity(
            base["entities"], anchor, po_operand, localizer_id
        ):
            rejected.append(v22._rejection(
                lexeme, localizer_id, "positional_po_wrapper",
                "positional_wrapper_would_skip_intervening_entity",
            ))
            continue
        path = list(anchor_paths[anchor["entity_id"]])
        path.extend(po_paths[v22._operand_key(po_operand)])
        path.extend(po_operand.get("dependency_path", []))
        evidence = v22._evidence_record(
            relation, lexeme, localizer_id, anchor, po_operand,
            "explicit_positional_po_wrapper", path,
        )
        v22._accept_if_valid(type_schema, evidence, accepted, rejected)
    return accepted, rejected


def _scope_anchor_entities(entities, scope_id, heads, dependency_relations):
    values = []
    paths = {}
    for entity in entities:
        head_id = entity.get("head_token_id")
        if head_id is None or int(head_id) >= scope_id:
            continue
        path = v23._path_to_anchor(
            int(head_id), {scope_id}, heads, dependency_relations,
            {"ATT"}, max_steps=4,
        )
        if path is None or not path:
            continue
        values.append(entity)
        paths[entity["entity_id"]] = path
    values = v23._unique_by_entity_id(values)
    if not values:
        return values, paths
    shortest = min(len(paths[item["entity_id"]]) for item in values)
    return [
        item for item in values
        if len(paths[item["entity_id"]]) == shortest
    ], paths


def _scope_links_inner(scope_id, inner, heads, dependency_relations):
    inner_anchor_id = int(inner["left"]["head_token_id"])
    direct = v23._path_to_anchor(
        scope_id, {inner_anchor_id}, heads, dependency_relations,
        {"ATT"}, max_steps=3,
    )
    if direct is not None and direct:
        return direct
    po_spine = _po_spine_anchor(inner["right"], heads)
    return v23._path_to_anchor(
        scope_id, {po_spine}, heads, dependency_relations,
        LOCALIZER_TO_SPINE_RELATIONS, max_steps=5,
    )


def _extract_outer_scopes(row, base, positional_added, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    accepted = []
    rejected = []
    for item in positional_added:
        inner = _relation_operand(item)
        inner_start = min(v23._operand_token_ids(inner))
        candidates = []
        for scope_id in range(1, int(item["trigger_token_id"])):
            lexeme = tokens[scope_id - 1]
            if lexeme not in OUTER_SCOPE_LEXEMES:
                continue
            anchors, anchor_paths = _scope_anchor_entities(
                base["entities"], scope_id, heads, dependency_relations
            )
            link_path = _scope_links_inner(
                scope_id, inner, heads, dependency_relations
            )
            if len(anchors) != 1 or link_path is None or not link_path:
                continue
            anchor = anchors[0]
            if int(anchor["token_span"][1]) + 1 >= scope_id:
                continue
            if scope_id >= inner_start:
                continue
            candidates.append((
                scope_id, lexeme, anchor,
                anchor_paths[anchor["entity_id"]] + link_path,
            ))
        if len(candidates) != 1:
            if candidates:
                rejected.append(v22._rejection(
                    item["lexeme"], item["trigger_token_id"],
                    "outer_scope_wrapper", "outer_scope_not_unique",
                ))
            continue
        scope_id, lexeme, anchor, path = candidates[0]
        path.extend(inner.get("dependency_path", []))
        evidence = v22._evidence_record(
            "IN", lexeme, scope_id, anchor, inner,
            "explicit_outer_scope_wrapper", path,
        )
        v22._accept_if_valid(type_schema, evidence, accepted, rejected)
    return accepted, rejected


def extract_relation_evidence(row, type_schema, enable_positional_wrapper=True,
                              enable_carrier_wrapper=True):
    base = v23.extract_relation_evidence(row, type_schema, enable_composition=True)
    if not enable_positional_wrapper and not enable_carrier_wrapper:
        result = dict(base)
        result["version"] = "v24_compatibility"
        return result

    added, added_rejections = ([], [])
    if enable_positional_wrapper:
        added, added_rejections = _extract_positional_wrappers(
            row, base, type_schema
        )
    carrier_added, carrier_rejections = ([], [])
    if enable_carrier_wrapper:
        carrier_base = dict(base)
        carrier_base["accepted"] = list(base["accepted"]) + list(added)
        carrier_added, carrier_rejections = _extract_carrier_wrappers(
            row, carrier_base, type_schema
        )
    outer_added, outer_rejections = _extract_outer_scopes(
        row, base, list(added) + list(carrier_added), type_schema
    )
    clause_ids = v23._clause_ids(list(row.get("tokens", [])))
    accepted = []
    seen = set()
    for item in list(base["accepted"]) + added + carrier_added + outer_added:
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
        list(base["rejected"]) + added_rejections + carrier_rejections
        + outer_rejections
    )
    result["version"] = "v24"
    return result
