#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V26 bottom-up parent closure over accepted V25 relation nodes.

V26 does not add a new L1 evidence family.  It uses explicit carrier,
copular, positional and scope structures to wrap already accepted V25 nodes
as L2/L3.  Parent construction is therefore conditioned on a real child
relation rather than on arbitrary entity-pair enumeration.
"""

from __future__ import print_function

try:
    from . import ltp_relation_evidence_v22 as v22
    from . import ltp_relation_evidence_v23 as v23
    from . import ltp_relation_evidence_v24 as v24
    from . import ltp_relation_evidence_v25 as v25
except (ImportError, ValueError):
    import ltp_relation_evidence_v22 as v22
    import ltp_relation_evidence_v23 as v23
    import ltp_relation_evidence_v24 as v24
    import ltp_relation_evidence_v25 as v25


PARENT_CHILD_RELATIONS = {"PO", "A", "BW", "C", "E", "W", "S", "N", "NE", "SE", "NW", "SW"}
V26_POSITIONAL_LOCALIZERS = dict(v24.POSITIONAL_LOCALIZERS)
V26_POSITIONAL_LOCALIZERS.update({
    # The original lexicon had 下段/下亚段 but omitted their upper and central
    # counterparts.  These are direct, symmetric stratigraphic localizers.
    "上段": "A",
    "上亚段": "A",
    "中亚段": "C",
})
OCCURRENCE_MARKERS = {"发育", "见", "可见", "中见", "出露", "常见", "含", "有", "属纯"}
IN_LOCALIZERS = {"中", "内", "内部", "附近", "范围内", "沟谷内", "带内"}
COPULAR_MARKER = "为"
LITHOLOGY_MARKER = "岩性"
FEATURE_MARKER = "特征"

load_type_schema = v25.load_type_schema
schema_allows = v25.schema_allows


def _entity_start(entity):
    return int(entity["token_span"][0]) + 1


def _entity_end(entity):
    return int(entity["token_span"][1]) + 1


def _tokens_between(tokens, left_end, right_start):
    if right_start <= left_end + 1:
        return []
    return tokens[left_end:right_start - 1]


def _tail_until_punctuation(tokens, start_id):
    values = []
    for token in tokens[start_id:]:
        if token in v22.CLAUSE_PUNCTUATION:
            break
        values.append(token)
    return values


def _contains_for_main(tokens):
    return "为主" in tokens or (
        COPULAR_MARKER in tokens and "主" in tokens
    )


def _relation_operand(item):
    return v22._relation_operand(
        item["relation"], item["lexeme"], item["trigger_token_id"],
        item["left"], item["right"], item["pattern"],
        item.get("dependency_path", []),
    )


def _child_entity_ids(item):
    return v24._operand_entity_ids(_relation_operand(item))


def _nearest_carriers_before(entities, child_item):
    child_start = min(v23._operand_token_ids(_relation_operand(child_item)))
    child_ids = _child_entity_ids(child_item)
    values = [
        entity for entity in entities
        if entity["entity_id"] not in child_ids
        and entity.get("head_token_id") is not None
        and _entity_end(entity) < child_start
    ]
    if not values:
        return []
    nearest_end = max(_entity_end(entity) for entity in values)
    return [entity for entity in values if _entity_end(entity) == nearest_end]


def _pair_hierarchy_link(tokens, left, right, heads,
                         dependency_relations, clause_ids):
    """Return one explicit dataset hierarchy link between two entities.

    This is intentionally narrower than a general ATT propagation rule.  A
    link needs both a surface marker and one dependency path.  The rule is
    used only while closing a parent around an already accepted child
    relation.
    """
    left_head = left.get("head_token_id")
    right_head = right.get("head_token_id")
    if left_head is None or right_head is None:
        return None
    if clause_ids[int(left_head) - 1] != clause_ids[int(right_head) - 1]:
        # A leading ``在地点，地层...`` phrase is an explicit location
        # scope even though punctuation gives it a separate clause id.  Do
        # not generalise this to ``沿/附近/以北``: those are independently
        # annotated directional or discourse modifiers in GNRED.
        trigger_id = _entity_start(left) - 1
        if (
            left.get("type") == "TM" and trigger_id >= 1
            and tokens[trigger_id - 1] == "在"
            and clause_ids[int(right_head) - 1]
            == clause_ids[int(left_head) - 1] + 1
        ):
            path = v24._dependency_path_between(
                int(left_head), int(right_head), heads,
                dependency_relations,
                v25.EXPLICIT_PATH_RELATIONS | {"FOB", "RAD"}, max_steps=8,
            )
            if path is not None:
                return (
                    "IN", "在", trigger_id,
                    "explicit_leading_location_scope", path,
                )
        return None
    between = _tokens_between(tokens, _entity_end(left), _entity_start(right))
    if not between or any(token in v22.CLAUSE_PUNCTUATION for token in between):
        return None

    candidates = []
    for token_id in range(_entity_end(left) + 1, _entity_start(right)):
        lexeme = tokens[token_id - 1]
        relation = V26_POSITIONAL_LOCALIZERS.get(lexeme)
        if relation in {"A", "BW", "C"}:
            candidates.append((relation, lexeme, token_id,
                               "explicit_carrier_positional_link"))
    if not candidates:
        for token_id in range(_entity_end(left) + 1, _entity_start(right)):
            lexeme = tokens[token_id - 1]
            if lexeme in {"中", "内", "内部", "之中"}:
                candidates.append(("IN", lexeme, token_id,
                                   "explicit_carrier_containment_link"))
    if not candidates and left.get("type") in {"TM", "GS"}:
        for token_id in range(_entity_end(left) + 1, _entity_start(right)):
            if tokens[token_id - 1] == "的":
                candidates.append(("IN", "的", token_id,
                                   "explicit_location_genitive_link"))
    if len(candidates) != 1:
        return None

    path = v24._dependency_path_between(
        int(left_head), int(right_head), heads, dependency_relations,
        v25.EXPLICIT_PATH_RELATIONS | {"FOB", "RAD"}, max_steps=8,
    )
    if path is None:
        return None
    relation, lexeme, trigger_id, pattern = candidates[0]
    return relation, lexeme, trigger_id, pattern, path


def _build_carrier_hierarchy(tokens, entities, carrier, child_item,
                             type_schema, heads, dependency_relations,
                             clause_ids, parent_relation):
    """Build the explicit left operand that terminates at ``carrier``.

    For example, ``地区中的层系中部组`` becomes
    ``C(IN(地区, 层系), 组)``.  Folding is left-to-right so no required
    intermediate node is skipped.
    """
    excluded = _child_entity_ids(child_item)
    ordered = sorted([
        entity for entity in entities
        if entity.get("entity_id") not in excluded
        and entity.get("head_token_id") is not None
        and _entity_start(entity) <= _entity_start(carrier)
    ], key=lambda entity: (_entity_start(entity), _entity_end(entity)))
    carrier_index = None
    for index, entity in enumerate(ordered):
        if entity.get("entity_id") == carrier.get("entity_id"):
            carrier_index = index
            break
    if carrier_index is None:
        return carrier, [], False

    selected = [carrier]
    links = []
    cursor = carrier
    for index in range(carrier_index - 1, -1, -1):
        previous = ordered[index]
        link = _pair_hierarchy_link(
            tokens, previous, cursor, heads, dependency_relations, clause_ids
        )
        if link is None:
            break
        selected.insert(0, previous)
        links.insert(0, link)
        cursor = previous

    if not links:
        return carrier, [], False

    # Pure containment associates outside an occurrence/positional parent:
    # ``X中的Y可见Z`` is IN(X, IN(Y, Z)), not IN(IN(X, Y), Z).
    # A positional link is different: ``X的Y顶部Z见R`` requires the complete
    # A(IN(X, Y), Z) carrier before IN can attach R.  Copular EQ likewise
    # consumes the complete explicit carrier chain.
    if (
        parent_relation != "EQ"
        and not any(link[0] in {"A", "BW", "C"} for link in links)
    ):
        return carrier, [], False

    operand = selected[0]
    evidence_items = []
    for right, link in zip(selected[1:], links):
        relation, lexeme, trigger_id, pattern, path = link
        evidence = v22._evidence_record(
            relation, lexeme, trigger_id, operand, right, pattern, path
        )
        accepted = []
        rejected = []
        v22._accept_if_valid(type_schema, evidence, accepted, rejected)
        if not accepted:
            return carrier, [], True
        evidence_items.extend(accepted)
        operand = _relation_operand(accepted[0])
    return operand, evidence_items, True


def _select_trigger(tokens, between, carrier_end, preferred):
    for token_id in range(carrier_end + 1, carrier_end + 1 + len(between)):
        if tokens[token_id - 1] in preferred:
            return token_id, tokens[token_id - 1]
    return None, None


def _classify_parent(tokens, carrier, child_item):
    child_operand = _relation_operand(child_item)
    child_tokens = v23._operand_token_ids(child_operand)
    child_start, child_end = min(child_tokens), max(child_tokens)
    between = _tokens_between(tokens, _entity_end(carrier), child_start)
    tail = _tail_until_punctuation(tokens, child_end)

    positional = []
    for offset, token in enumerate(between, _entity_end(carrier) + 1):
        relation = V26_POSITIONAL_LOCALIZERS.get(token)
        if relation in {"A", "BW", "C"}:
            positional.append((offset, token, relation))
    if len(positional) == 1:
        token_id, lexeme, relation = positional[0]
        return relation, lexeme, token_id, "explicit_child_positional_parent"

    # ``X岩性以 child 为特征`` is an identity/lithology statement in GNRED.
    if (
        LITHOLOGY_MARKER in between and "以" in between
        and FEATURE_MARKER in tail and COPULAR_MARKER in tail
    ):
        token_id, lexeme = _select_trigger(
            tokens, between, _entity_end(carrier), {"以"}
        )
        return "EQ", lexeme, token_id, "explicit_lithology_feature_parent"

    # A direct copular statement maps to EQ.  ``岩性为`` is deliberately
    # excluded because GNRED consistently treats that construction as IN in
    # this corpus; ``为以...为特征`` is also not a direct copular statement.
    if (
        COPULAR_MARKER in between and LITHOLOGY_MARKER not in between
        and "以" not in between
        and not any(token in OCCURRENCE_MARKERS for token in between)
    ):
        token_id, lexeme = _select_trigger(
            tokens, between, _entity_end(carrier), {COPULAR_MARKER}
        )
        return "EQ", lexeme, token_id, "explicit_copular_child_parent"

    # Explicit containment/occurrence evidence has priority over generic
    # copular wording.
    in_markers = OCCURRENCE_MARKERS | IN_LOCALIZERS
    if any(token in in_markers for token in between):
        token_id, lexeme = _select_trigger(
            tokens, between, _entity_end(carrier), in_markers
        )
        return "IN", lexeme, token_id, "explicit_occurrence_child_parent"

    if LITHOLOGY_MARKER in between and COPULAR_MARKER in between:
        token_id, lexeme = _select_trigger(
            tokens, between, _entity_end(carrier), {COPULAR_MARKER}
        )
        return "IN", lexeme, token_id, "explicit_lithology_content_parent"

    if "以" in between and _contains_for_main(tail):
        token_id, lexeme = _select_trigger(
            tokens, between, _entity_end(carrier), {"以"}
        )
        return "IN", lexeme, token_id, "explicit_as_main_child_parent"

    if (
        COPULAR_MARKER in between and "以" in between
        and FEATURE_MARKER in tail
    ):
        token_id, lexeme = _select_trigger(
            tokens, between, _entity_end(carrier), {COPULAR_MARKER}
        )
        return "IN", lexeme, token_id, "explicit_nested_feature_parent"

    # A named location phrase followed by an exposed/intercalated child is a
    # surface-explicit carrier even when LTP attaches both modifiers to the
    # first rock rather than to the occurrence verb.
    if carrier.get("text", "").endswith(("地区", "一带")):
        combined = between + tail
        if any(token in {"出露", "发育", "见", "可见"} for token in combined):
            preferred = {"出露", "发育", "见", "可见"}
            token_id, lexeme = _select_trigger(
                tokens, between, _entity_end(carrier), preferred
            )
            if token_id is None:
                for token_id in range(child_end + 1, child_end + 1 + len(tail)):
                    if tokens[token_id - 1] in preferred:
                        lexeme = tokens[token_id - 1]
                        break
            return "IN", lexeme, token_id, "explicit_named_location_parent"
    return None


def _extract_child_parents(row, base, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    clause_ids = v23._clause_ids(tokens)
    accepted = []
    rejected = []
    for item in base["accepted"]:
        if item.get("relation") not in PARENT_CHILD_RELATIONS:
            continue
        if int((item.get("node") or {}).get("depth", 0)) != 1:
            continue
        carriers = _nearest_carriers_before(base["entities"], item)
        if len(carriers) != 1:
            continue
        carrier = carriers[0]
        classified = _classify_parent(tokens, carrier, item)
        if classified is None:
            continue
        relation, lexeme, trigger_id, pattern = classified
        if trigger_id is None or lexeme is None:
            continue
        child_clause = clause_ids[int(item["trigger_token_id"]) - 1]
        carrier_clause = clause_ids[int(carrier["head_token_id"]) - 1]
        if carrier_clause != child_clause:
            # Cross-clause copular and positional attachment is precisely the
            # old window-style failure mode.  Retain only an explicit named
            # location scope for IN; its marker must occur before the child.
            if (
                relation != "IN" or carrier.get("type") != "TM"
                or not carrier.get("text", "").endswith(
                    ("地区", "一带", "附近", "北部", "南部")
                )
            ):
                continue
        carrier_operand, hierarchy_items, had_hierarchy = (
            _build_carrier_hierarchy(
                tokens, base["entities"], carrier, item, type_schema,
                heads, dependency_relations, clause_ids, relation,
            )
        )
        if had_hierarchy and not hierarchy_items:
            rejected.append(v22._rejection(
                lexeme, trigger_id, pattern,
                "explicit_carrier_hierarchy_schema_disallowed",
            ))
            continue
        child = _relation_operand(item)
        path = v24._dependency_path_between(
            int(carrier["head_token_id"]), int(item["trigger_token_id"]),
            heads, dependency_relations,
            v25.EXPLICIT_PATH_RELATIONS | {"FOB"}, max_steps=8,
        )
        if path is None:
            rejected.append(v22._rejection(
                lexeme, trigger_id, pattern,
                "explicit_child_parent_dependency_path_missing",
            ))
            continue
        path = list(path) + list(item.get("dependency_path", []))
        evidence = v22._evidence_record(
            relation, lexeme, trigger_id, carrier_operand, child, pattern, path
        )
        parent_accepted = []
        v22._accept_if_valid(
            type_schema, evidence, parent_accepted, rejected
        )
        if parent_accepted:
            accepted.extend(hierarchy_items)
            accepted.extend(parent_accepted)
    return accepted, rejected


def _nearest_outer_before(entities, inner_item):
    inner = _relation_operand(inner_item)
    inner_ids = v24._operand_entity_ids(inner)
    inner_start = min(v23._operand_token_ids(inner))
    values = [
        entity for entity in entities
        if entity["entity_id"] not in inner_ids
        and entity.get("head_token_id") is not None
        and _entity_end(entity) < inner_start
    ]
    if not values:
        return []
    nearest_end = max(_entity_end(entity) for entity in values)
    return [entity for entity in values if _entity_end(entity) == nearest_end]


def _extract_explicit_hierarchical_scopes(row, base, l2_items, type_schema):
    tokens = list(row.get("tokens", []))
    heads, dependency_relations, unused_children = v22._dependency_maps(row)
    accepted = []
    rejected = []
    for item in list(base["accepted"]) + list(l2_items):
        if int((item.get("node") or {}).get("depth", 0)) != 2:
            continue
        if item.get("left", {}).get("kind") == "relation":
            continue
        outers = _nearest_outer_before(base["entities"], item)
        if len(outers) != 1:
            continue
        outer = outers[0]
        inner_carrier = item["left"]
        between = _tokens_between(
            tokens, _entity_end(outer), _entity_start(inner_carrier)
        )
        inner_to_child = _tokens_between(
            tokens, _entity_end(inner_carrier),
            min(v23._operand_token_ids(item["right"]))
        )

        trigger_id = None
        lexeme = None
        pattern = None
        if "中" in between and "的" in between:
            trigger_id = _entity_end(outer) + 1 + between.index("中")
            lexeme = "中"
            pattern = "explicit_containment_hierarchical_scope"
        elif (
            "的" in between
            and any(token in {"南段", "北段", "上段", "下段", "中段"} for token in between)
            and "剖面" in inner_to_child and "中" in inner_to_child
        ):
            trigger_id = _entity_end(inner_carrier) + 1 + inner_to_child.index("中")
            lexeme = "中"
            pattern = "explicit_section_profile_scope"
        if trigger_id is None:
            continue

        inner = _relation_operand(item)
        path = v24._dependency_path_between(
            int(outer["head_token_id"]), int(item["trigger_token_id"]),
            heads, dependency_relations,
            v25.EXPLICIT_PATH_RELATIONS | {"FOB", "RAD"}, max_steps=10,
        )
        if path is None:
            rejected.append(v22._rejection(
                lexeme, trigger_id, pattern,
                "explicit_hierarchical_scope_dependency_path_missing",
            ))
            continue
        evidence = v22._evidence_record(
            "IN", lexeme, trigger_id, outer, inner, pattern,
            list(path) + list(item.get("dependency_path", [])),
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


def extract_relation_evidence(row, type_schema, enable_parent_closure=True):
    base = v25.extract_relation_evidence(row, type_schema, enable_explicit_l1=True)
    if not enable_parent_closure:
        result = dict(base)
        result["version"] = "v26_compatibility"
        return result

    l2, l2_rejections = _extract_child_parents(row, base, type_schema)
    clause_ids = v23._clause_ids(list(row.get("tokens", [])))
    annotated_l2 = [
        item if item.get("node") else v23._annotate_evidence(item, clause_ids)
        for item in l2
    ]
    l3, l3_rejections = _extract_explicit_hierarchical_scopes(
        row, base, annotated_l2, type_schema
    )

    accepted = []
    seen = set()
    for item in list(base["accepted"]) + annotated_l2 + l3:
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
    result["rejected"] = list(base["rejected"]) + l2_rejections + l3_rejections
    result["version"] = "v26"
    return result
