#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared zero-shot prompt and response schema for candidate scoring."""

import json
import re


PROMPT_VERSION = "nested_candidate_ltp_compare_v9"


# Dataset convention: for every directional relation R(A, B), B is located
# relative to A. PO is symmetric; EC/DC follow the dataset semantics below.
RELATION_DESC = {
    "EQ": "EQ(A, B)：原句明确表达 A 是 B、A 属于 B，或 A 的岩性、建造、性质/特征为 B。",
    "EC": "EC(A, B)：原句明确表达 A 与 B 接触、相接、相连或相邻；本数据关系也包括整合接触与不整合接触。",
    "DC": "DC(A, B)：原句明确表达 A 与 B 之间、二者分隔或二者构成区间的空间关系。",
    "IN": "IN(A, B)：B 位于、分布、出现或包含在 A 内；A 是位置或容器，B 是其中的对象。",
    "PO": "PO(A, B)：A 与 B 共生、伴生、互层、夹杂、交错、相交或并存；该关系不区分方向。",
    "A": "A(A, B)：B 位于 A 的上部、顶部或上方。",
    "BW": "BW(A, B)：B 位于 A 的下部、底部或下方。",
    "C": "C(A, B)：B 位于 A 的中部、中间或轴部。",
    "E": "E(A, B)：B 位于 A 的东部、东侧或以东。",
    "W": "W(A, B)：B 位于 A 的西部、西侧或以西。",
    "S": "S(A, B)：B 位于 A 的南部、南侧或以南。",
    "N": "N(A, B)：B 位于 A 的北部、北侧或以北。",
    "NE": "NE(A, B)：B 位于 A 的东北部、东北侧或东北方向。",
    "SE": "SE(A, B)：B 位于 A 的东南部、东南侧或东南方向。",
    "NW": "NW(A, B)：B 位于 A 的西北部、西北侧或西北方向。",
    "SW": "SW(A, B)：B 位于 A 的西南部、西南侧或西南方向。",
}


def _entity_payloads(group):
    payloads = []
    ids_by_text = {}
    types_by_text = {}
    for index, entity in enumerate(group.get("entities", [])):
        entity_id = "E{}".format(index)
        text = str(entity.get("text", ""))
        payload = {
            "entity_id": entity_id,
            "text": text,
            "type": entity.get("type"),
            "char_span_inclusive": entity.get("span"),
        }
        if entity.get("token_span") is not None:
            payload["token_span_inclusive"] = entity.get("token_span")
        payloads.append(payload)
        ids_by_text.setdefault(text, []).append(entity_id)
        types_by_text.setdefault(text, set()).add(
            str(entity.get("type", "<UNKNOWN>"))
        )
    return payloads, ids_by_text, types_by_text


def _node_types(node, types_by_text):
    if isinstance(node, dict):
        return {str(node.get("rid", node.get("relation", node.get("rel"))))}
    return set(types_by_text.get(str(node), {"<UNKNOWN>"}))


def _schema_index(group):
    result = {}
    for item in group.get("schema_checks", []):
        key = (
            str(item.get("relation")),
            tuple(sorted(str(value) for value in item.get("left_operand_types", []))),
            tuple(sorted(str(value) for value in item.get("right_operand_types", []))),
        )
        result[key] = bool(item.get("allowed_by_schema"))
    return result


def _candidate_tree(node, ids_by_text, types_by_text, schema_by_signature):
    if not isinstance(node, dict):
        text = str(node)
        matching_ids = ids_by_text.get(text, [])
        entity = {"text": text}
        if len(matching_ids) == 1:
            entity["entity_id"] = matching_ids[0]
        else:
            entity["matching_entity_ids"] = matching_ids
        return {"entity": entity}

    relation = str(node.get("rid", node.get("relation", node.get("rel"))))
    left = node.get("subj")
    right = node.get("obj")
    left_types = tuple(sorted(_node_types(left, types_by_text)))
    right_types = tuple(sorted(_node_types(right, types_by_text)))
    signature = (relation, left_types, right_types)
    if signature not in schema_by_signature:
        raise ValueError("candidate schema check is missing: {!r}".format(signature))
    return {
        "relation": relation,
        "schema_check": {
            "left_operand_types": list(left_types),
            "right_operand_types": list(right_types),
            "allowed": schema_by_signature[signature],
        },
        "left": _candidate_tree(
            left, ids_by_text, types_by_text, schema_by_signature
        ),
        "right": _candidate_tree(
            right, ids_by_text, types_by_text, schema_by_signature
        ),
    }


def model_visible_sentence_groups(sentence_groups):
    """Expose evidence and final trees, but not V21 construction provenance."""
    result = []
    for group in sentence_groups:
        required = {
            "sentence",
            "entities",
            "tokens",
            "dependency_edges",
            "rule_hypotheses_may_be_wrong",
            "schema_checks",
            "candidates",
        }
        missing = sorted(required - set(group))
        if missing:
            raise ValueError(
                "model-facing group {} lacks v5 evidence: {}".format(
                    group.get("sid"), missing
                )
            )
        entities, ids_by_text, types_by_text = _entity_payloads(group)
        schema_by_signature = _schema_index(group)
        candidates = []
        for candidate in group.get("candidates", []):
            candidates.append({
                "task_id": str(candidate["task_id"]),
                "candidate_tree": _candidate_tree(
                    candidate["triple"],
                    ids_by_text,
                    types_by_text,
                    schema_by_signature,
                ),
            })
        result.append({
            "sid": str(group.get("sid")),
            "sentence": group.get("sentence", ""),
            "entities": entities,
            "tokens": group.get("tokens", []),
            "dependency_edges": group.get("dependency_edges", []),
            "rule_hypotheses": group.get(
                "rule_hypotheses_may_be_wrong", []
            ),
            "candidates": candidates,
        })
    return result


def _build_full_prompt_v7(sentence_groups):
    relation_lines = ["- {}".format(RELATION_DESC[key]) for key in RELATION_DESC]
    return """你是一名中文地质嵌套关系审核专家。

请判断每个candidate_tree是否构成了对原句、LTP结构和规则证据的合理、完整嵌套解释，并给出0到1之间的confidence。该分数将用于候选之间的置信度比较和阈值筛选；你的任务不是脱离输入证据重新抽取关系。

confidence表示完整candidate_tree获得直接支持的程度，而不是LTP或规则是否产生过这个候选，也不是某个关键词、单个实体、局部子关系或大致语义正确的程度。

【关系定义】
{relation_desc}

【审核依据】

1. LTP分词、词性和依存结构是定位关系线索、论元连接和嵌套顺序的主要结构证据。它们解释候选的结构来源，但候选由LTP构造出来本身不能直接证明该候选正确。
2. rule_hypotheses是基于LTP和领域规则得到的重要关系线索。必须检查candidate_tree中的每一层是否准确使用了相应的关系类型、论元、方向和组合顺序；同一条规则线索不能无差别地证明多个不同的竞争结构。
3. 原句用于理解并核对LTP和规则线索。高分要求原句语义与结构证据共同对完整候选提供正向支持；仅仅没有发现明显冲突、只匹配到关系词或只支持局部子关系，不能获得高分。若原句与某种结构解释存在重要歧义，应降低到中间分数，而不是默认相信任一候选。
4. candidate_tree中每层schema_check都是结构约束。allowed=false时该候选不能成立；allowed=true是必要条件，但不能单独证明当前嵌套正确。
5. 必须严格按照关系定义检查每一层的left和right，不能因日常语言中关系大致合理而翻转论元方向或替换关系类型。若某个必要子关系明确不成立，整棵候选树不能获得高分。
6. 同一句可以有零个、一个或多个正确候选。评分时应同时比较该句全部候选：共享相同实体和结构线索，但关系方向、关系类型或嵌套位置不同的候选属于竞争解释。应给完整证据支持更充分、逐层组合更一致的候选更高分；无法由证据区分的竞争候选应保留在中间分数，不能全部获得近似的极高分。明确表达的子关系作为独立候选时，仍可与包含它的正确完整树同时获得高分。
7. 不新增、删除、修改或重新构造候选，不使用输入之外的地质知识补全关系。

【confidence规则】

- 接近1：候选的关系类型、方向、论元和每层嵌套均得到原句、LTP结构和规则线索的直接支持，且明显优于相关竞争解释。对这种完整且清晰的候选，不要因过度保守而压低分数。
- 接近0.5：候选具有一定依据，但只有局部关系得到支持，或者嵌套顺序、论元归属、关系类型及竞争解释仍存在重要歧义。
- 接近0：候选明显误用了LTP或规则证据，或者关系类型、方向、论元、嵌套顺序与主要证据明确冲突。

confidence必须是0到1之间的连续数值。高分会使候选在后续比较中更有竞争力，因此不能仅因候选来自LTP或符合schema就给高分；但完整证据明确支持的候选应获得足够高的分数。0.9以上和0.1以下只用于证据近乎无歧义或存在明确冲突的少数候选，不要把0.01、0.98等极端值作为常用评分模板。存在合理歧义或竞争解释时应向0.5收缩，并根据支持程度使用有区分度的连续分数。

【输入说明】

- sentence：原句。
- entities：实体文本、类型及位置。
- tokens：LTP分词、token编号和词性。
- dependency_edges：LTP依存边，dependent依赖于head。
- rule_hypotheses：基于LTP和规则得到的关系线索。
- candidates：需要审核的候选列表。每个候选只包含task_id和最终candidate_tree，不包含候选的生成过程。
- candidate_tree中的relation、left和right表示完整嵌套结构；schema_check表示该层左右操作数类型及其schema合法性。

【输出要求】

只输出一个JSON对象，不输出解释、Markdown或其他文字。每个输入task_id必须且只能输出一次，并保持输入顺序。

输出格式：
{{
  "results": [
    {{
      "task_id": "原task_id",
      "confidence": 0.5
    }}
  ]
}}

【待审核输入】
{groups_json}
""".format(
        relation_desc="\n".join(relation_lines),
        groups_json=json.dumps(
            model_visible_sentence_groups(sentence_groups),
            ensure_ascii=False,
            indent=2,
        ),
    )


def build_full_prompt(sentence_groups):
    relation_lines = ["- {}".format(RELATION_DESC[key]) for key in RELATION_DESC]
    return """你是一名中文地质嵌套关系审核专家。

请依据原句、LTP结构和规则线索，审核每个candidate_tree能否作为该句合理、完整的关系解释，并给出0到1之间的confidence。你的任务是审核输入候选，不是脱离输入重新抽取或改写关系。

confidence表示完整candidate_tree作为最终关系解释的可信程度，后续会用于候选之间的分数比较。评分对象是整棵树，不是某个关键词、局部子关系或“候选是否由LTP生成”。

【关系定义】
{relation_desc}

【审核方法】
1. LTP分词、词性、依存结构和rule_hypotheses是判断论元连接与嵌套顺序的主要结构依据，应优先用于理解候选。不要无理由否定这些证据，但候选来自LTP或规则本身也不等于整棵树必然正确。
2. 逐层检查candidate_tree的关系类型、left/right方向、论元归属和嵌套顺序，并用原句语义核对这种结构解释。schema_check.allowed=true只说明论元类型允许；allowed=false时该层不能成立。
3. 对同一sid中的全部候选先进行内部比较，再分别评分。共享相同LTP或规则线索的候选不应机械地获得相同高分；若它们在关系类型、方向或嵌套位置上互相竞争，应让证据更完整、更一致的候选得分更高，难以区分时保留中间分数。互不冲突且分别得到明确支持的多个候选可以同时获得较高分。
4. 局部子关系成立不能单独证明完整树成立。必要层存在明显歧义时，完整树应降低到中间分数；但同一子关系作为独立候选时应按它自身的证据评分。
5. 不使用输入之外的地质常识补全关系，不新增、删除、修改或重新构造候选。

【confidence校准】
- 0.50到0.70：整棵候选树总体得到直接正向支持，各层关系、方向、论元和嵌套顺序没有关键错误，但支持强度尚不足以进入高分。
- 高于0.70：仅适用于整棵候选树的关系类型、方向、论元和嵌套顺序均得到非常清晰、一致的支持，且不存在明显竞争解释的情况。
- 0.30到低于0.50：只有部分关系或局部结构得到支持，尚不能证明完整candidate_tree成立。这类候选应低于0.50，不要因为“有一定依据”而默认给0.50。
- 低于0.3：关键关系类型、方向、论元或嵌套顺序与主要证据冲突。
- confidence应使用有区分度的连续数值。即使候选非常清晰，通常也优先使用0.70到0.80表达，原则上不要超过0.85；即使候选明显错误，通常也可用0.10到0.30表达。不要把0.85以上和0.05以下等极端值当作常用评分模板，也不要反复使用某一个接近区间上限的固定分数。同句候选应根据实际证据差异拉开分数，而不是把原来的极端模板替换成新的固定模板。

评分前请在内部依次完成“逐层核对—同句比较—分数校准”，但不要输出分析过程。confidence必须是有区分度的连续数值。

【输入说明】
- sentence：原句。
- entities：实体文本、类型及位置。
- tokens和dependency_edges：LTP结构证据。
- rule_hypotheses：基于LTP和规则得到的关系线索。
- candidates：待审核候选；candidate_tree中的relation、left和right表示完整嵌套结构，schema_check表示该层的论元类型约束。

【输出要求】
只输出一个JSON对象，不输出解释、Markdown或其他文字。每个输入task_id必须且只能输出一次，并保持输入顺序。
输出格式：
{{
  "results": [
    {{
      "task_id": "原task_id",
      "confidence": 0.5
    }}
  ]
}}

【待审核输入】
{groups_json}
""".format(
        relation_desc="\n".join(relation_lines),
        groups_json=json.dumps(
            model_visible_sentence_groups(sentence_groups),
            ensure_ascii=False,
            indent=2,
        ),
    )


def build_user_prompt(sentence_groups):
    """Compatibility alias for callers migrated to the one-message prompt."""
    return build_full_prompt(sentence_groups)


def build_web_prompt(sentence_groups):
    """Return the exact one-message prompt used by both web and API."""
    return build_full_prompt(sentence_groups)


def extract_results(text):
    """Parse the single supported response schema, allowing a code fence."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("response must be a JSON object containing a results array")
    return data["results"]
