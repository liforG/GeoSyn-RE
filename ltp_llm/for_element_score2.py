# -*- coding: utf-8 -*-
import argparse
import json
import os
import unicodedata


# ===================== Sentence normalization =====================
def normalize_sentence(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("－", "-").replace("–", "-").replace("—", "-")
    text = text.replace("′", "'").replace("“", '"').replace("”", '"')
    text = text.replace(" ", "")
    return text
# =========================================================


def load_test_json(test_file_path):
    data = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_triples_jsonl(triples_file_path):
    triples_by_sentence = {}
    triples_by_sid = {}
    with open(triples_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            norm_sent = normalize_sentence(item['sentence'])
            triples_by_sentence[norm_sent] = item
            sid = item.get("sid")
            if sid is not None:
                triples_by_sid[str(sid)] = item
    return triples_by_sentence, triples_by_sid


def extract_entity_from_sentence(sentence, index_pair):
    s, e = index_pair
    return sentence[s:e + 1]


# ===================== Entity lookup =====================
def _index_from_entity_id(entity_id, entity, test_entities):
    if entity_id is None:
        return -1
    value = str(entity_id)
    if not value.startswith("e") or not value[1:].isdigit():
        raise ValueError("invalid candidate entity_id: {}".format(value))
    index = int(value[1:])
    if index < 0 or index >= len(test_entities):
        raise ValueError("candidate entity_id out of range: {}".format(value))
    if normalize_sentence(entity) != normalize_sentence(test_entities[index]):
        raise ValueError(
            "candidate entity_id/text mismatch: {} {!r} != {!r}".format(
                value, entity, test_entities[index]
            )
        )
    return index


def find_entity_index(entity, entity_map, test_entities, entity_id=None):
    if not isinstance(entity, str):
        return -1

    explicit_index = _index_from_entity_id(entity_id, entity, test_entities)
    if explicit_index >= 0:
        return explicit_index

    normalized_entity = normalize_sentence(entity)
    exact = entity_map.get(normalized_entity, [])
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            "ambiguous repeated entity text without entity_id: {!r}".format(entity)
        )

    compatible = []
    for i, e in enumerate(test_entities):
        normalized_test_entity = normalize_sentence(e)
        if normalized_entity in normalized_test_entity or normalized_test_entity in normalized_entity:
            compatible.append(i)
    if len(compatible) == 1:
        return compatible[0]
    if len(compatible) > 1:
        raise ValueError(
            "ambiguous substring entity match without entity_id: {!r}".format(entity)
        )

    return -1
# ==================================================


# ===================== Stable recursive structure =====================
def get_triple_subject(node):
    return node.get("subject", node.get("subj"))


def get_triple_relation(node):
    return node.get("relation", node.get("rid", node.get("rel")))


def get_triple_object(node):
    return node.get("object", node.get("obj"))


def resolve_structure(node, entity_map, test_entities, entity_id=None):
    """Resolve a candidate tree to entity-index based immutable structure."""
    if isinstance(node, str):
        entity_index = find_entity_index(
            node, entity_map, test_entities, entity_id=entity_id
        )
        if entity_index < 0:
            return None
        return ("ENTITY", entity_index)
    if not isinstance(node, dict):
        return None

    subj = get_triple_subject(node)
    rel = get_triple_relation(node)
    obj = get_triple_object(node)
    evidence = node.get("evidence") or {}
    left = resolve_structure(
        subj, entity_map, test_entities,
        entity_id=evidence.get("left_entity_id") if isinstance(subj, str) else None,
    )
    right = resolve_structure(
        obj, entity_map, test_entities,
        entity_id=evidence.get("right_entity_id") if isinstance(obj, str) else None,
    )
    if rel is None or left is None or right is None:
        return None
    return ("RELATION", str(rel), left, right)


def structure_layer(structure):
    if structure[0] == "ENTITY":
        return 0
    return 1 + max(structure_layer(structure[2]), structure_layer(structure[3]))


def structure_to_json(structure):
    if structure[0] == "ENTITY":
        return {"entity_index": structure[1]}
    return {
        "relation": structure[1],
        "left": structure_to_json(structure[2]),
        "right": structure_to_json(structure[3]),
    }


def structure_sort_key(structure):
    return json.dumps(
        structure_to_json(structure),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def collect_scored_structures(triples_info, entity_map, test_entities):
    """Return unique explicit candidate structures and their final scores."""
    score_by_structure = {}
    for triple in triples_info.get("triples", []):
        structure = resolve_structure(triple, entity_map, test_entities)
        if structure is None:
            continue
        confidence = float(triple.get("confidence", 0.5))
        previous = score_by_structure.get(structure)
        if previous is not None and abs(previous - confidence) > 1e-12:
            raise ValueError(
                "duplicate candidate structure has conflicting confidence: {} vs {}".format(
                    previous, confidence
                )
            )
        score_by_structure[structure] = confidence
    return score_by_structure


def ensure_element(structure, elements, structure_index, score_by_structure):
    """Recursively materialize one stable structure in legacy element format."""
    if structure[0] == "ENTITY":
        return int(structure[1])
    if structure in structure_index:
        return structure_index[structure]

    left_index = ensure_element(
        structure[2], elements, structure_index, score_by_structure
    )
    right_index = ensure_element(
        structure[3], elements, structure_index, score_by_structure
    )
    confidence = float(score_by_structure.get(structure, 0.0))
    elements.append([structure[1], [left_index, right_index], confidence])
    index = len(elements) - 1
    structure_index[structure] = index
    return index
# ==================================================


def convert_triples_to_fusion(test_item, triples_info, sentence):
    """Build legacy elements plus order-independent recursive LLM scores."""
    elements = list(test_item['entity'])
    test_entities = [
        extract_entity_from_sentence(sentence, e[1])
        for e in test_item['entity']
    ]
    entity_map = {}
    for index, entity in enumerate(test_entities):
        entity_map.setdefault(normalize_sentence(entity), []).append(index)
    score_by_structure = collect_scored_structures(
        triples_info, entity_map, test_entities
    )
    ordered_structures = sorted(
        score_by_structure,
        key=lambda item: (structure_layer(item), structure_sort_key(item)),
    )
    structure_index = {}
    for structure in ordered_structures:
        ensure_element(structure, elements, structure_index, score_by_structure)

    llm_scores = [
        {
            "structure": structure_to_json(structure),
            "confidence": score_by_structure[structure],
            "layer": structure_layer(structure),
        }
        for structure in ordered_structures
    ]
    return {
        "elements": elements,
        "llm_scores": llm_scores,
    }


def convert_triples_to_elements(test_item, triples_info, sentence):
    """Backward-compatible entry point used by existing audits and callers."""
    return convert_triples_to_fusion(test_item, triples_info, sentence)["elements"]


def create_dataset_with_scores(test_data, triples_by_sentence, triples_by_sid, output_file_path, missing_output_path=None):
    ok, miss = 0, 0
    missing_rows = []

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            sent = item['sentence']
            norm_sent = normalize_sentence(sent)
            sid = str(item.get("sid"))
            triples_info = triples_by_sid.get(sid) or triples_by_sentence.get(norm_sent)

            if triples_info is None:
                miss += 1
                missing_rows.append({
                    "sid": item.get("sid"),
                    "sentence": sent,
                    "normalized_sentence": norm_sent,
                })
                f.write(json.dumps({
                    "sid": item["sid"],
                    "entity": item["entity"],
                    "element": item["entity"],
                    "llm_score_format": "recursive_entity_index_v1",
                    "llm_scores": [],
                    "sentence": sent
                }, ensure_ascii=False) + "\n")
                continue

            fusion = convert_triples_to_fusion(item, triples_info, sent)

            f.write(json.dumps({
                "sid": item["sid"],
                "entity": item["entity"],
                "element": fusion["elements"],
                "llm_score_format": "recursive_entity_index_v1",
                "llm_scores": fusion["llm_scores"],
                "sentence": sent
            }, ensure_ascii=False) + "\n")

            ok += 1

    if missing_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(missing_output_path)), exist_ok=True)
        with open(missing_output_path, "w", encoding="utf-8") as f:
            json.dump(missing_rows, f, ensure_ascii=False, indent=2)

    return ok, miss


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Convert scored triples to element-score JSONL for LLM fusion.")
    parser.add_argument("--test", default="/data/disk2/sw/data/5-1/test.json",
                        help="Input test JSONL path.")
    parser.add_argument("--triples", default=os.path.join(base, "outputs/triples_scored_updated.jsonl"),
                        help="Input triples_scored JSONL path.")
    parser.add_argument("--output", default=os.path.join(base, "outputs/final_all_2.jsonl"),
                        help="Output final_all_2 JSONL path.")
    parser.add_argument("--missing-output", default=None,
                        help="Optional JSON path for unmatched test sentences.")
    args = parser.parse_args()

    test_file = args.test
    triples_file = args.triples
    output_file = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    test_data = load_test_json(test_file)
    triples_by_sentence, triples_by_sid = load_triples_jsonl(triples_file)

    ok, miss = create_dataset_with_scores(
        test_data, triples_by_sentence, triples_by_sid, output_file, args.missing_output
    )

    print("完成")
    print("成功:", ok)
    print("未匹配:", miss)
    print("输出:", output_file)
    if args.missing_output:
        print("未匹配明细:", args.missing_output)


if __name__ == "__main__":
    main()
