# nested_pipeline.py
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict, Any, Tuple, Optional

base_dir = os.path.dirname(os.path.abspath(__file__))
fenxi_dir = os.path.join(base_dir, "fenxi")
cache_dir = os.path.join(base_dir, ".jieba_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault("JIEBA_TMPDIR", cache_dir)
print("[RUNNING FILE]:", __file__)

import jieba
from pyltp import Postagger, Parser

REL_MAP = {
    "EQ": ["是", "为", "属于", "属"],
    "IN": ["含",  "包括", "位于", "在", "由", "构成", "组成", "包含", "分布", "处于", "出现", "出现于", "出现在", "中", "之中", "内部", "内"],
    "PO": ["伴", "交错", "穿插", "穿越", "叠置", "交汇", "相交", "夹杂", "夹", "夹有", "互层"],
    "A":  ["上", "上部", "之上", "顶部"],
    "BW": ["下", "下部", "之下", "底部", "下伏", "下段", "下亚段"],
    "C":  ["中部", "中间", "轴部", "中段"],
    "N":  ["北", "北部", "以北", "北侧", "北缘"],
    "S":  ["南", "南部", "以南", "南侧", "南缘"],
    "E":  ["东", "东部", "以东", "东侧", "东缘"],
    "W":  ["西", "西部", "以西", "西侧", "西缘"],
    "NE": ["东北", "东北部", "东北侧"],
    "SE": ["东南", "东南部", "东南侧"],
    "NW": ["西北", "西北部", "西北侧"],
    "SW": ["西南", "西南部", "西南侧"],
}
LEX_TO_REL = {lex: rel for rel, lexs in REL_MAP.items() for lex in lexs}

TRIGGER_POS = {"v", "vd", "vn", "vi", "vg", "a", "nd", "p"}
SUBJECT_CHILD_REL_DEFAULT = {"SBV"}
PARENT_OBJECT_RELS = {"ATT", "VOB", "FOB", "POB", "CMP", "ADV"}
OBJECT_CHILD_RELS = {"VOB", "FOB", "POB"}

pos_model_path = os.path.join(fenxi_dir, "pos.model")
par_model_path = os.path.join(fenxi_dir, "parser.model")

_POSTAGGER: Optional[Postagger] = None
_PARSER: Optional[Parser] = None


def ensure_models_loaded() -> None:
    global _POSTAGGER, _PARSER
    if _POSTAGGER is None:
        _POSTAGGER = Postagger()
        _POSTAGGER.load(pos_model_path)
    if _PARSER is None:
        _PARSER = Parser()
        _PARSER.load(par_model_path)


def release_models() -> None:
    global _POSTAGGER, _PARSER
    if _POSTAGGER:
        _POSTAGGER.release()
        _POSTAGGER = None
    if _PARSER:
        _PARSER.release()
        _PARSER = None


_GEO_DIRECTION_ND = {
    "轴部","核部"
}
_GEO_DIRECTION_ND_SUFFIX = tuple(_GEO_DIRECTION_ND)

_GEO_NOUN_SUFFIXES = (
    "背斜", "向斜", "断层", "断裂", "断裂带",
    "褶皱", "褶皱带", "穹窿",
    "岩体", "岩群", "岩组", "岩层",
    "组", "系", "段", "统", "群"
)
_GEO_NOUN_SINGLE = {
    "背斜", "向斜", "断层", "断裂", "褶皱"
}

def _fix_geo_pos(tokens: List[str], pos: List[str]) -> List[str]:
    fixed = list(pos)
    for i, t in enumerate(tokens):
        p = fixed[i]

        if t in _GEO_DIRECTION_ND or any(t.endswith(suf) for suf in _GEO_DIRECTION_ND_SUFFIX):
            if p != "nd":
                fixed[i] = "nd"
            continue
        if t in _GEO_NOUN_SINGLE or any(t.endswith(suf) for suf in _GEO_NOUN_SUFFIXES):
            if p.startswith('v') or p in {'a'}:
                fixed[i] = 'n'
            continue

        if len(t) >= 2 and (t[0] in "FfKkPp") and any(ch.isdigit() for ch in t[1:]):
            if p.startswith('v'):
                fixed[i] = 'n'
            continue

    return fixed


def pyltp_parse(sentence: str, extra_words: Optional[List[str]] = None) -> Dict[str, List]:
    ensure_models_loaded()
    if extra_words:
        for w in extra_words:
            if w:
                jieba.add_word(w)
    tokens = list(jieba.cut(sentence))
    pos = list(_POSTAGGER.postag(tokens))
    pos = _fix_geo_pos(tokens, pos)
    arcs = _PARSER.parse(tokens, pos)
    dep = [(i + 1, arc.head, arc.relation) for i, arc in enumerate(arcs)]
    return {"tokens": tokens, "pos": pos, "dep": dep}


def align_entities(tokens: List[str], sentence: str, entities: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    offsets = []
    idx = 0
    for tok in tokens:
        start = sentence.find(tok, idx)
        end = start + len(tok)
        offsets.append((start, end))
        idx = end
    mapping = {i: [] for i in range(len(tokens))}
    for ent in entities:
        span = ent.get("span")
        if not span:
            continue
        s, e = span
        e2 = e + 1
        covered = [i for i, (ts, te) in enumerate(offsets) if not (te <= s or ts >= e2)]
        if not covered:
            continue
        enriched = {**ent, "token_span": (covered[0], covered[-1])}
        for i in covered:
            mapping[i].append(enriched)
    return mapping


def rel_from_lexeme(word: str, pos: str) -> str:
    if word in LEX_TO_REL:
        return LEX_TO_REL[word]
    if pos == "nd":
        if word.startswith("北"): return "N"
        if word.startswith("南"): return "S"
        if word.startswith("东"): return "E"
        if word.startswith("西"): return "W"
        if word in {"中", "内部", "内"}: return "IN"
        return "IN"
    if pos == "p":
        return "IN"
    if word in {"出现", "出现在", "出现于", "分布", "位于", "处于"}:
        return "IN"
    return "IN"


def collect_triggers(parsed: Dict[str, Any], entity_map: Dict[int, List[Dict[str, Any]]]) -> List[Dict]:
    tokens, pos_tags, dep = parsed["tokens"], parsed["pos"], parsed["dep"]
    id2token = {i + 1: {"w": tokens[i], "p": pos_tags[i], "head": dep[i][1], "rel": dep[i][2]} for i in range(len(tokens))}

    triggers = []

    for idx, meta in id2token.items():
        if meta["p"] not in TRIGGER_POS:
            continue
        trig_word = meta["w"]
        trig_pos = meta["p"]
        rel_to_head = meta["rel"]

        subj_rels = {"SBV"}
        if trig_pos in {"nd", "p"}:
            subj_rels.add("ATT")

        subjects = []
        for j, m in id2token.items():
            if m["head"] == idx and m["rel"] in subj_rels:
                subjects.extend(entity_map.get(j - 1, []))
        objects = []
        head_idx = meta["head"]
        if head_idx > 0 and rel_to_head in {"ATT", "VOB", "FOB", "POB", "CMP", "ADV"}:
            objects.extend(entity_map.get(head_idx - 1, []))

        for j, m in id2token.items():
            if m["head"] == idx and m["rel"] in {"VOB", "FOB", "POB"}:
                objects.extend(entity_map.get(j - 1, []))

        subj_text = [s["text"] for s in subjects]
        obj_text = [o["text"] for o in objects]

        rel_label = rel_from_lexeme(trig_word, trig_pos)

        if subj_text or obj_text:
            triggers.append({
                "trigger": trig_word,
                "pos": trig_pos,
                "rel": rel_label,
                "subjects": subj_text,
                "objects": obj_text
            })

    return triggers


def run_pipeline(sentence: str, entities: List[Dict[str, Any]]) -> List[Dict]:
    parsed = pyltp_parse(sentence, [e["text"] for e in entities])
    entity_map = align_entities(parsed["tokens"], sentence, entities)
    return collect_triggers(parsed, entity_map)


if __name__ == "__main__":
    sentence = "齐耀山断层（F1）：分布于齐耀山背斜轴部中桥-郑家垭一带中，长约 70 km。"
    entities = [
        {"text": "齐耀山断层", "span": [0, 4], "type": "GS"},
        {"text": "齐耀山背斜", "span": [13, 17], "type": "GS"},
        {"text": "中桥-郑家垭一带", "span": [20, 27], "type": "TM"},
    ]

    results = run_pipeline(sentence, entities)
    print(f"\n句子：{sentence}\n")
    for trig in results:
        print(f"[TRIGGER] {trig['trigger']}/{trig['pos']} → 主语={trig['subjects']} 宾语={trig['objects']}")
    print("\n—— 结束 ——\n")
    release_models()
