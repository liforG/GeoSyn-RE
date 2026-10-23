# -*- coding: utf-8 -*-
"""Inference entry point for isolated model-vs-LLM compare fusion."""

from __future__ import print_function

import argparse
import json
import os

import torch as t
from transformers import BertTokenizer

from config import Config
from data_utils import get_data_loader2, get_type_info, load_pllm_scores_jsonl
from inference_llm_ronghe import infer_on_loader, load_checkpoint
from models_compare import ITransformerCompare
from utils import compute_metric


def build_compare_model(cfg, all_type):
    return ITransformerCompare(
        cfg.bert_dir,
        cfg.max_sent_len,
        cfg.bert_hidden,
        cfg.bert_head,
        cfg.transformer_layer,
        all_type,
        cfg.trans_type,
        runtime_config=cfg,
    ).to(cfg.device)


def _entity_structure(entity):
    return ("ENTITY", entity[0], tuple(entity[1]))


def _resolve(relations, entities):
    structures = [_entity_structure(entity) for entity in entities]
    layers = [0] * len(entities)
    resolved = []
    for relation, operands in relations:
        op1, op2 = int(operands[0]), int(operands[1])
        structure = ("RELATION", relation, structures[op1], structures[op2])
        layer = 1 + max(layers[op1], layers[op2])
        structures.append(structure)
        layers.append(layer)
        resolved.append((structure, layer))
    return resolved


def _gold_relations(elements, entities):
    entity_count = len(entities)
    if list(elements[:entity_count]) == list(entities):
        return elements[entity_count:]
    return elements


def _empty_stats():
    return {
        "gold": 0,
        "baseline_predictions": 0,
        "compare_predictions": 0,
        "baseline_correct": 0,
        "compare_correct": 0,
        "promoted": 0,
        "promoted_correct": 0,
        "promoted_wrong": 0,
        "demoted": 0,
        "demoted_correct": 0,
        "demoted_wrong": 0,
        "replacements": 0,
        "corrected_replacements": 0,
        "harmed_replacements": 0,
        "gold_to_gold_replacements": 0,
        "wrong_to_wrong_replacements": 0,
    }


def build_flip_audit(sids, elements, entities, baseline, compare):
    overall = _empty_stats()
    per_layer = {}
    records = []

    def layer_stats(layer):
        key = str(layer)
        if key not in per_layer:
            per_layer[key] = _empty_stats()
        return per_layer[key]

    def add_counts(stats, gold_set, base_set, compare_set):
        promoted = compare_set - base_set
        demoted = base_set - compare_set
        stats["gold"] += len(gold_set)
        stats["baseline_predictions"] += len(base_set)
        stats["compare_predictions"] += len(compare_set)
        stats["baseline_correct"] += len(base_set & gold_set)
        stats["compare_correct"] += len(compare_set & gold_set)
        stats["promoted"] += len(promoted)
        stats["promoted_correct"] += len(promoted & gold_set)
        stats["promoted_wrong"] += len(promoted - gold_set)
        stats["demoted"] += len(demoted)
        stats["demoted_correct"] += len(demoted & gold_set)
        stats["demoted_wrong"] += len(demoted - gold_set)

        base_by_pair = {(item[2], item[3]): item for item in base_set}
        compare_by_pair = {(item[2], item[3]): item for item in compare_set}
        for pair in set(base_by_pair) & set(compare_by_pair):
            before = base_by_pair[pair]
            after = compare_by_pair[pair]
            if before == after:
                continue
            stats["replacements"] += 1
            before_gold = before in gold_set
            after_gold = after in gold_set
            if not before_gold and after_gold:
                stats["corrected_replacements"] += 1
            elif before_gold and not after_gold:
                stats["harmed_replacements"] += 1
            elif before_gold and after_gold:
                stats["gold_to_gold_replacements"] += 1
            else:
                stats["wrong_to_wrong_replacements"] += 1

    for sid, gold_elements, sample_entities, base_relations, compare_relations in zip(
        sids, elements, entities, baseline, compare
    ):
        gold_resolved = _resolve(
            _gold_relations(gold_elements, sample_entities), sample_entities
        )
        base_resolved = _resolve(base_relations, sample_entities)
        compare_resolved = _resolve(compare_relations, sample_entities)
        gold_set = set(item[0] for item in gold_resolved)
        base_set = set(item[0] for item in base_resolved)
        compare_set = set(item[0] for item in compare_resolved)
        add_counts(overall, gold_set, base_set, compare_set)

        layers = set(item[1] for item in gold_resolved + base_resolved + compare_resolved)
        for layer in layers:
            layer_gold = set(item[0] for item in gold_resolved if item[1] == layer)
            layer_base = set(item[0] for item in base_resolved if item[1] == layer)
            layer_compare = set(item[0] for item in compare_resolved if item[1] == layer)
            add_counts(layer_stats(layer), layer_gold, layer_base, layer_compare)

        for change, changed in (
            ("promoted", compare_set - base_set),
            ("demoted", base_set - compare_set),
        ):
            for structure in sorted(changed, key=repr):
                records.append({
                    "sid": str(sid),
                    "change": change,
                    "is_gold": structure in gold_set,
                    "relation_structure": structure,
                })

    for stats in [overall] + list(per_layer.values()):
        stats["net_predictions"] = (
            stats["compare_predictions"] - stats["baseline_predictions"]
        )
        stats["net_correct"] = (
            stats["compare_correct"] - stats["baseline_correct"]
        )
    return {"overall": overall, "per_layer": per_layer, "records": records}


def print_audit(audit):
    def line(label, stats):
        print(
            "[COMPARE] {} predictions={}->{} | correct={}->{} | "
            "promoted={} (correct={}, wrong={}) | demoted={} "
            "(correct={}, wrong={}) | replacements={} "
            "(corrected={}, harmed={})".format(
                label,
                stats["baseline_predictions"],
                stats["compare_predictions"],
                stats["baseline_correct"],
                stats["compare_correct"],
                stats["promoted"],
                stats["promoted_correct"],
                stats["promoted_wrong"],
                stats["demoted"],
                stats["demoted_correct"],
                stats["demoted_wrong"],
                stats["replacements"],
                stats["corrected_replacements"],
                stats["harmed_replacements"],
            )
        )

    line("Overall", audit["overall"])
    for layer in sorted(audit["per_layer"], key=int):
        line("L{}".format(layer), audit["per_layer"][layer])


def metric_from_counts(correct, predicted, gold):
    precision = float(correct) / predicted if predicted else 0.0
    recall = float(correct) / gold if gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": int(correct),
        "predicted": int(predicted),
        "gold": int(gold),
    }


def strict_metrics_from_audit(audit):
    stats = audit["overall"]
    return {
        "baseline": metric_from_counts(
            stats["baseline_correct"], stats["baseline_predictions"], stats["gold"]
        ),
        "compare": metric_from_counts(
            stats["compare_correct"], stats["compare_predictions"], stats["gold"]
        ),
    }


def load_prediction_rows(path):
    """Load plain relation arrays or save_predict JSONL records."""
    with open(path, "r", encoding="utf-8") as stream:
        content = stream.read().strip()
    if not content:
        return []
    try:
        value = json.loads(content)
    except ValueError:
        value = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not isinstance(value, list):
        raise ValueError("reference predictions must be a JSON array or JSONL arrays")
    predictions = []
    sids = []
    has_record_sids = None
    for index, row in enumerate(value):
        is_record = isinstance(row, dict)
        if has_record_sids is None:
            has_record_sids = is_record
        elif has_record_sids != is_record:
            raise ValueError("reference predictions mix records and relation arrays")

        if not is_record:
            if not isinstance(row, list):
                raise ValueError("reference prediction row {} is not an array".format(index))
            predictions.append(row)
            continue

        entities = row.get("entity")
        elements = row.get("element")
        if not isinstance(entities, list) or not isinstance(elements, list):
            raise ValueError(
                "reference record {} must contain entity/element arrays".format(index)
            )
        if len(elements) < len(entities):
            raise ValueError("reference record {} has truncated elements".format(index))
        predictions.append(elements[len(entities):])
        sids.append(str(row.get("sid")))
    return predictions, (sids if has_record_sids else None)


def canonical_json_value(value):
    if isinstance(value, (list, tuple)):
        return [canonical_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="figure-6-5/geosyn-re-1/model.pt")
    parser.add_argument("--test", default="data/gnred/test.json")
    parser.add_argument("--data-dir", default="data/gnred")
    parser.add_argument("--type-file", default="data/type3.json")
    parser.add_argument("--radical-vocab-file", default="data/gnred/train.json")
    parser.add_argument(
        "--bert-path",
        help="BERT directory used by the checkpoint; defaults to Config.bert_dir.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sent-len", type=int, default=128)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--pllm-file")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="new/outputs/gnred_v29_expanded_v1/compare")
    parser.add_argument("--equivalence-only", action="store_true")
    parser.add_argument(
        "--reference-predictions",
        help="Optional saved model-only predictions that the empty-LLM run must reproduce.",
    )
    parser.add_argument("--audit-flips", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("isolated compare currently requires --batch-size 1")
    if not args.equivalence_only and not args.pllm_file:
        raise ValueError("--pllm-file is required unless --equivalence-only is used")

    root = os.path.dirname(os.path.abspath(__file__))
    cfg = Config()
    cfg.data_dir = os.path.abspath(os.path.join(root, args.data_dir))
    cfg.type_file = os.path.abspath(os.path.join(root, args.type_file))
    cfg.radical_vocab_file = os.path.abspath(
        os.path.join(root, args.radical_vocab_file)
    )
    if args.bert_path:
        cfg.bert_dir = os.path.abspath(os.path.join(root, args.bert_path))
    cfg.max_sent_len = args.max_sent_len
    cfg.device = args.device
    model_path = os.path.abspath(os.path.join(root, args.model))
    test_file = os.path.abspath(os.path.join(root, args.test))
    output_dir = os.path.abspath(os.path.join(root, args.output_dir))

    tokenizer = BertTokenizer.from_pretrained(cfg.bert_dir)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    all_type, relations = get_type_info(cfg.type_file)
    loader = get_data_loader2(
        cfg.test_prefix,
        cfg.data_dir,
        tokenizer=tokenizer,
        batch_size=1,
        max_seq_len=cfg.max_sent_len,
        data_file=test_file,
    )
    # One model instance is used for both passes.  ``ITransformerCompare`` now
    # delegates inference directly to the baseline layer-wise implementation;
    # compare changes only the current candidate score calculation.
    model = load_checkpoint(build_compare_model(cfg, all_type), model_path, cfg.device)
    baseline, elements, entities, sids, _ = infer_on_loader(
        model,
        loader,
        relations,
        all_type,
        cfg.device,
        pad_token_id,
        mode="scratch",
        pllm_scores_map={},
        threshold=args.threshold,
        alpha=1.0,
        # Match the project's actual model-only baseline invocation exactly:
        # weighted fusion, alpha=1, and an empty PLLM map.
        fusion_mode="weighted",
        llm_threshold=0.0,
        max_layer=args.max_layer,
    )

    if args.equivalence_only:
        compare_empty, empty_elements, empty_entities, empty_sids, _ = infer_on_loader(
            model,
            loader,
            relations,
            all_type,
            cfg.device,
            pad_token_id,
            mode="scratch",
            pllm_scores_map={},
            threshold=args.threshold,
            alpha=1.0,
            fusion_mode="compare",
            llm_threshold=0.0,
            max_layer=args.max_layer,
        )
        aligned = (
            sids == empty_sids
            and elements == empty_elements
            and entities == empty_entities
        )
        identical = aligned and baseline == compare_empty
        print(
            "[EQUIVALENCE] identical={} | model_predictions={} | "
            "compare_empty_predictions={}".format(
                identical,
                sum(len(item) for item in baseline),
                sum(len(item) for item in compare_empty),
            )
        )
        if not identical:
            raise RuntimeError("empty compare changed model-only predictions")
        if args.reference_predictions:
            reference_path = os.path.abspath(
                os.path.join(root, args.reference_predictions)
            )
            reference_rows, reference_sids = load_prediction_rows(reference_path)
            reference = canonical_json_value(reference_rows)
            current = canonical_json_value(baseline)
            sid_identical = reference_sids is None or reference_sids == sids
            reference_identical = sid_identical and current == reference
            print(
                "[REFERENCE] identical={} | sid_alignment={} | "
                "reference_predictions={} | model_predictions={}".format(
                    reference_identical,
                    sid_identical,
                    sum(len(item) for item in reference),
                    sum(len(item) for item in current),
                )
            )
            if not reference_identical:
                raise RuntimeError(
                    "current model-only predictions do not match the supplied reference"
                )
        precision, recall, f1 = compute_metric(elements, baseline, entities)
        print(
            "[BASELINE] Precision={:.5f} | Recall={:.5f} | F1={:.5f}".format(
                precision, recall, f1
            )
        )
        strict_baseline = strict_metrics_from_audit(
            build_flip_audit(sids, elements, entities, baseline, baseline)
        )["baseline"]
        print(
            "[BASELINE-STRICT] correct={} predicted={} gold={} | "
            "P={:.5f} R={:.5f} F1={:.5f}".format(
                strict_baseline["correct"],
                strict_baseline["predicted"],
                strict_baseline["gold"],
                strict_baseline["precision"],
                strict_baseline["recall"],
                strict_baseline["f1"],
            )
        )
        return

    pllm_path = os.path.abspath(os.path.join(root, args.pllm_file))
    pllm_scores = load_pllm_scores_jsonl(pllm_path)
    compare, compare_elements, compare_entities, compare_sids, fusion_stats = infer_on_loader(
        model,
        loader,
        relations,
        all_type,
        cfg.device,
        pad_token_id,
        mode="scratch",
        pllm_scores_map=pllm_scores,
        threshold=args.threshold,
        alpha=1.0,
        fusion_mode="compare",
        llm_threshold=0.0,
        max_layer=args.max_layer,
    )
    if sids != compare_sids or elements != compare_elements or entities != compare_entities:
        raise RuntimeError("model and compare inference are not aligned")

    precision, recall, f1 = compute_metric(elements, compare, entities)
    baseline_p, baseline_r, baseline_f1 = compute_metric(elements, baseline, entities)
    audit = build_flip_audit(sids, elements, entities, baseline, compare)
    strict_metrics = strict_metrics_from_audit(audit)
    print(
        "[RESULT] baseline P={} R={} F1={} | compare P={} R={} F1={}".format(
            baseline_p, baseline_r, baseline_f1, precision, recall, f1
        )
    )
    print_audit(audit)
    print(
        "[STRICT] baseline P={:.5f} R={:.5f} F1={:.5f} | "
        "compare P={:.5f} R={:.5f} F1={:.5f}".format(
            strict_metrics["baseline"]["precision"],
            strict_metrics["baseline"]["recall"],
            strict_metrics["baseline"]["f1"],
            strict_metrics["compare"]["precision"],
            strict_metrics["compare"]["recall"],
            strict_metrics["compare"]["f1"],
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predict.json"), "w", encoding="utf-8") as stream:
        json.dump(compare, stream, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as stream:
        json.dump(
            {
                "threshold": args.threshold,
                "baseline": {
                    "precision": baseline_p,
                    "recall": baseline_r,
                    "f1": baseline_f1,
                    "predicted": sum(len(item) for item in baseline),
                },
                "compare": {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "predicted": sum(len(item) for item in compare),
                },
                "pllm_sids": len(pllm_scores),
                "strict_complete_tree": strict_metrics,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(output_dir, "compare_audit.json"), "w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "runtime_compare.json"), "w", encoding="utf-8") as stream:
        json.dump(fusion_stats, stream, ensure_ascii=False, indent=2)
    print("[INFO] output={}".format(output_dir))


if __name__ == "__main__":
    main()
