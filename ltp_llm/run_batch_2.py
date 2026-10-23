# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = '/data/disk2/sw/data/5-1/test.json'
OUT_DIR = os.path.join(BASE_DIR, "outputs")
TOK_DIR = os.path.join(OUT_DIR, "tokens")
POS_DIR = os.path.join(OUT_DIR, "pos")
DEP_DIR = os.path.join(OUT_DIR, "deps")
ENRICHED_PATH = os.path.join(OUT_DIR, "enriched.jsonl")

os.environ.setdefault("LTP_POS_MODEL", os.path.join(BASE_DIR, "fenxi", "pos.model"))
os.environ.setdefault("LTP_PARSER_MODEL", os.path.join(BASE_DIR, "fenxi", "parser.model"))

import nested_pipeline as NP

def ensure_dirs(out_dir):
    for d in [out_dir, os.path.join(out_dir, "tokens"), os.path.join(out_dir, "pos"), os.path.join(out_dir, "deps")]:
        os.makedirs(d, exist_ok=True)

def preprocess_sentence(sentence: str) -> str:
    if not sentence:
        return ""
    sentence = sentence.replace("－", "-")
    sentence = sentence.replace("—", "-")
    return sentence

def parse_entities_from_line(line_obj: Dict[str, Any], processed_sentence: str) -> List[Dict[str, Any]]:
    ents_raw = line_obj.get("entity") or []
    entities: List[Dict[str, Any]] = []
    sentence = processed_sentence or ""

    for item in ents_raw:
        try:
            etype, span = item[0], item[1]
            s, e = int(span[0]), int(span[1])
            text = sentence[s:e + 1]
            if text:
                entities.append({"text": text, "span": [s, e], "type": etype})
        except Exception:
            continue

    return entities

def write_simple_files(out_dir: str, sid: str, tokens: List[str], pos: List[str], dep_triplets: List):
    tok_dir = os.path.join(out_dir, "tokens")
    pos_dir = os.path.join(out_dir, "pos")
    dep_dir = os.path.join(out_dir, "deps")

    with open(os.path.join(tok_dir, f"{sid}.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))

    with open(os.path.join(pos_dir, f"{sid}.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(pos))

    conllu_path = os.path.join(dep_dir, f"{sid}.conllu")
    with open(conllu_path, "w", encoding="utf-8") as f:
        for i, (idx, head, rel) in enumerate(dep_triplets):
            f.write(f"{idx}\t_\t_\t{head}\t{rel}\n")

def main():
    parser = argparse.ArgumentParser(description="Batch run nested_pipeline on JSONL.")
    parser.add_argument("--input", "-i", default=DEFAULT_IN, help="输入 JSONL（每行一条）")
    parser.add_argument("--out-dir", default=OUT_DIR, help="LTP 解析结果输出目录")
    parser.add_argument("--output", "-o", default=None, help="富集输出 jsonl 路径；默认写入 <out-dir>/enriched.jsonl")
    parser.add_argument("--pos-model", default=os.environ.get("LTP_POS_MODEL", os.path.join(BASE_DIR, "fenxi", "pos.model")),
                        help="pyltp 词性模型路径")
    parser.add_argument("--parser-model", default=os.environ.get("LTP_PARSER_MODEL", os.path.join(BASE_DIR, "fenxi", "parser.model")),
                        help="pyltp 依存句法模型路径")
    parser.add_argument("--silent", action="store_true", help="静默模式（屏蔽内核打印）")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    output_path = os.path.abspath(args.output) if args.output else os.path.join(out_dir, "enriched.jsonl")
    ensure_dirs(out_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    NP.pos_model_path = args.pos_model
    NP.par_model_path = args.parser_model

    import builtins
    real_print = builtins.print
    if args.silent:
        builtins.print = lambda *a, **k: None

    NP.ensure_models_loaded()

    n_ok, n_bad = 0, 0

    with open(args.input, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:

        for raw in tqdm(fin, desc="Processing"):
            raw = raw.strip()
            if not raw:
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                n_bad += 1
                continue

            sid = str(obj.get("sid", "NA"))
            sentence = preprocess_sentence(obj.get("sentence", ""))
            entities = parse_entities_from_line(obj, sentence)

            try:
                parsed = NP.pyltp_parse(sentence, [e["text"] for e in entities])
                entity_map = NP.align_entities(parsed["tokens"], sentence, entities)
                triggers = NP.collect_triggers(parsed, entity_map)

                write_simple_files(out_dir, sid, parsed["tokens"], parsed["pos"], parsed["dep"])

                enriched = {
                    "sid": sid,
                    "sentence": sentence,
                    "entities": entities,
                    "tokens": parsed["tokens"],
                    "pos": parsed["pos"],
                    "dep": parsed["dep"],
                    "entity_map": entity_map,
                    "triggers": triggers
                }

                fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                n_ok += 1

            except Exception as e:
                n_bad += 1
                enriched_err = {
                    "sid": sid,
                    "sentence": sentence,
                    "error": repr(e)
                }
                fout.write(json.dumps(enriched_err, ensure_ascii=False) + "\n")

    NP.release_models()

    if args.silent:
        builtins.print = real_print

    print(f"完成：成功 {n_ok} 条，失败 {n_bad} 条；富集输出 -> {output_path}")

if __name__ == "__main__":
    main()
