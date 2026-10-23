# -*- coding: utf-8 -*-

import os
import json
import argparse

import torch as t
from transformers import BertTokenizer

from config import Config
from data_utils import get_data_loader2, get_type_info, load_pllm_scores_jsonl
from utils import compute_metric
from models import ITransformer



PLLM_FILE = "/data/disk2/sw/new/outputs/final_all_2.jsonl"
THRESHOLD = 0.5
ALPHA = 0.1


def build_model(cfg, all_type):
    model = ITransformer(
        cfg.bert_dir,
        cfg.max_sent_len,
        cfg.bert_hidden,
        cfg.bert_head,
        cfg.transformer_layer,
        all_type,
        cfg.trans_type,
        runtime_config=cfg,
    ).to(cfg.device)
    return model


def load_checkpoint(model, model_path, device):
    state = t.load(model_path, map_location=device)
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    if isinstance(state, dict):
        model.validate_radical_vocab_metadata(state.get("radical_vocab"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def infer_on_loader(
    model,
    test_iter,
    relations,
    all_type,
    device,
    pad_token_id,
    mode="scratch",
    pllm_scores_map=None,
    threshold=0.5,
    alpha=0.7,
    fusion_mode="weighted",
    llm_threshold=0.95,
    max_layer=6,
):
    all_predict, all_elem, all_entity, all_sid = [], [], [], []
    fusion_stats = []
    use_guided = (mode.lower() == "guided")

    with t.no_grad():
        for batch in test_iter:
            sentence = batch["ids"]
            if not t.is_tensor(sentence):
                sentence = t.tensor(sentence, dtype=t.long)
            sentence = sentence.to(device)

            sentence_len = batch.get("sent_len")
            if sentence_len is None:
                sentence_len = (sentence != pad_token_id).sum(dim=1).tolist()
            elif t.is_tensor(sentence_len):
                sentence_len = sentence_len.tolist()

            entity = batch["entity"]
            element = batch["element"]
            entity_type = [[e[0] for e in se] for se in entity]
            entity_index = [[e[1] for e in se] for se in entity]

            x_mask = (sentence != pad_token_id).long()

            if use_guided:
                out = model.predict_guided(
                    x=sentence,
                    x_mask=x_mask,
                    x_lens=sentence_len,
                    entity_index=entity_index,
                    entity_type=entity_type,
                    relations=relations,
                    all_type=all_type
                )
                predict = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
            else:
                sid_list = batch.get("sid", None)

                if sid_list is None:
                    pllm_batch = [{} for _ in range(sentence.shape[0])]
                else:
                    if t.is_tensor(sid_list):
                        sid_list = sid_list.tolist()
                    sid_list = [str(s) for s in sid_list]

                    if pllm_scores_map is None:
                        pllm_batch = [{} for _ in sid_list]
                    else:
                        pllm_batch = [pllm_scores_map.get(s, {}) for s in sid_list]

                predict = model.predict_scratch(
                    x=sentence,
                    x_mask=x_mask,
                    entity_index=entity_index,
                    entity_type=entity_type,
                    relations=relations,
                    threshold=threshold,
                    pllm_scores=pllm_batch,
                    alpha=alpha,
                    fusion_mode=fusion_mode,
                    llm_threshold=llm_threshold,
                    max_layer=max_layer,
                )
                fusion_stats.extend(getattr(model, "last_llm_fusion_stats", []))

            all_predict.extend(predict)
            all_elem.extend(element)
            all_entity.extend(entity)
            sid_list = batch.get("sid", [])
            if t.is_tensor(sid_list):
                sid_list = sid_list.tolist()
            all_sid.extend([str(sid) for sid in sid_list])

    return all_predict, all_elem, all_entity, all_sid, fusion_stats


def relation_representations_by_layer(elements, entity):
    """Resolve index-based relations into stable recursive representations."""
    entity_num = len(entity)
    if list(elements[:entity_num]) == list(entity):
        relations = elements[entity_num:]
    else:
        relations = elements

    resolved = list(entity)
    resolved_layers = [0] * entity_num
    by_layer = {}
    for relation in relations:
        rel, operands = relation
        op1, op2 = int(operands[0]), int(operands[1])
        if op1 >= len(resolved) or op2 >= len(resolved):
            raise ValueError(
                "relation operand index exceeds resolved element list: {} {}".format(
                    relation, len(resolved)
                )
            )
        representation = (rel, (resolved[op1][1], resolved[op2][1]))
        layer = 1 + max(resolved_layers[op1], resolved_layers[op2])
        resolved.append(representation)
        resolved_layers.append(layer)
        by_layer.setdefault(layer, set()).add(representation)
    return by_layer


def audit_prediction_flips(sids, elements, entities, baseline_preds, fused_preds):
    """Compare final alpha=1 and fused prediction sets against Gold."""
    if not (
        len(sids) == len(elements) == len(entities)
        == len(baseline_preds) == len(fused_preds)
    ):
        raise ValueError("flip audit inputs are not aligned")

    summary = {}
    records = []

    def layer_stats(layer):
        if layer not in summary:
            summary[layer] = {
                "baseline_predictions": 0,
                "fused_predictions": 0,
                "retained": 0,
                "promoted": 0,
                "promoted_gold": 0,
                "promoted_wrong": 0,
                "demoted": 0,
                "demoted_gold": 0,
                "demoted_wrong": 0,
            }
        return summary[layer]

    for sid, gold, entity, baseline, fused in zip(
        sids, elements, entities, baseline_preds, fused_preds
    ):
        gold_by_layer = relation_representations_by_layer(gold, entity)
        baseline_by_layer = relation_representations_by_layer(baseline, entity)
        fused_by_layer = relation_representations_by_layer(fused, entity)
        layers = set(gold_by_layer) | set(baseline_by_layer) | set(fused_by_layer)

        for layer in layers:
            gold_set = gold_by_layer.get(layer, set())
            baseline_set = baseline_by_layer.get(layer, set())
            fused_set = fused_by_layer.get(layer, set())
            promoted = fused_set - baseline_set
            demoted = baseline_set - fused_set
            retained = baseline_set & fused_set
            stats = layer_stats(layer)
            stats["baseline_predictions"] += len(baseline_set)
            stats["fused_predictions"] += len(fused_set)
            stats["retained"] += len(retained)

            for change, relations in (("promoted", promoted), ("demoted", demoted)):
                for relation in sorted(relations, key=repr):
                    is_gold = relation in gold_set
                    stats[change] += 1
                    stats[change + ("_gold" if is_gold else "_wrong")] += 1
                    records.append({
                        "sid": sid,
                        "layer": layer,
                        "change": change,
                        "gold": is_gold,
                        "relation": relation,
                    })

    overall = layer_stats("overall")
    for layer, stats in list(summary.items()):
        if layer == "overall":
            continue
        for key, value in stats.items():
            overall[key] += value
    overall["net_prediction_change"] = overall["promoted"] - overall["demoted"]
    overall["net_correct_change"] = overall["promoted_gold"] - overall["demoted_gold"]
    for layer, stats in summary.items():
        if layer == "overall":
            continue
        stats["net_prediction_change"] = stats["promoted"] - stats["demoted"]
        stats["net_correct_change"] = stats["promoted_gold"] - stats["demoted_gold"]

    ordered_layers = {
        str(layer): summary[layer]
        for layer in sorted(key for key in summary if key != "overall")
    }
    return {"overall": overall, "per_layer": ordered_layers, "records": records}


def print_flip_audit(audit):
    def print_stats(label, stats):
        print(
            "[FLIP] {} baseline={} | fused={} | retained={} | "
            "promoted={} (gold={}, wrong={}) | "
            "demoted={} (gold={}, wrong={}) | "
            "net_predictions={:+d} | net_correct={:+d}".format(
                label,
                stats["baseline_predictions"],
                stats["fused_predictions"],
                stats["retained"],
                stats["promoted"],
                stats["promoted_gold"],
                stats["promoted_wrong"],
                stats["demoted"],
                stats["demoted_gold"],
                stats["demoted_wrong"],
                stats["net_prediction_change"],
                stats["net_correct_change"],
            )
        )

    print_stats("Overall", audit["overall"])
    for layer, stats in audit["per_layer"].items():
        print_stats("L{}".format(layer), stats)


def main(fusion_mode="weighted"):
    parser = argparse.ArgumentParser(
        description="ITransformer inference (scratch/guided) + LLM {} fusion".format(
            fusion_mode
        )
    )
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (model.pt).")
    parser.add_argument("--test", type=str, default=None, help="Path to test json.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dataset directory used by the checkpoint.")
    parser.add_argument("--type-file", type=str, default=None,
                        help="Schema file used for candidate generation.")
    parser.add_argument("--radical-vocab-file", type=str, default=None,
                        help="Training JSON used to rebuild the checkpoint radical vocabulary.")
    parser.add_argument("--device", type=str, default=None, help="Inference device, for example cuda:0.")
    parser.add_argument("--max-sent-len", type=int, default=None,
                        help="Maximum sentence length used by the checkpoint.")
    parser.add_argument("--max-layer", type=int, default=3,
                        help="Maximum nested relation layer used during inference.")
    parser.add_argument("--pllm-file", type=str, default=PLLM_FILE, help="Path to final_all_2.jsonl LLM score file.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Prediction threshold for scratch mode.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="LLM fusion weight for scratch mode.")
    parser.add_argument(
        "--llm-threshold",
        type=float,
        default=0.95,
        help="DeepSeek admission threshold used by add mode.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for predict.json. Defaults to checkpoint dir.")
    parser.add_argument("--debug-fusion", action="store_true", help="Print LLM fusion hit statistics.")
    parser.add_argument(
        "--audit-flips",
        action="store_true",
        help="Run an alpha=1 baseline and audit exact final prediction changes.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--mode", type=str, default="scratch", choices=["scratch", "guided"],
                        help="Inference mode: 'scratch' or 'guided'.")
    args = parser.parse_args()

    cfg = Config()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.type_file is not None:
        cfg.type_file = args.type_file
    if args.radical_vocab_file is not None:
        cfg.radical_vocab_file = args.radical_vocab_file
    if args.device is not None:
        cfg.device = args.device
    if args.max_sent_len is not None:
        cfg.max_sent_len = args.max_sent_len
    device = cfg.device
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_dir)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model_path = args.model or os.path.join(cfg.model_dir, "1126", "model.pt")
    test_file = args.test or os.path.join(cfg.data_dir, "5-1", "test.json")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {model_path}")
    print(f"[INFO] Test file: {test_file}")
    print(f"[INFO] Data directory: {cfg.data_dir}")
    print(f"[INFO] Type file: {cfg.type_file}")
    print(f"[INFO] Radical vocabulary file: {cfg.radical_vocab_file}")
    print(f"[INFO] Maximum sentence length: {cfg.max_sent_len}")
    print(f"[INFO] Maximum relation layer: {args.max_layer}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Threshold: {args.threshold}")
    print(f"[INFO] Fusion mode: {fusion_mode}")
    if fusion_mode == "weighted":
        print(f"[INFO] Alpha: {args.alpha}")
    elif fusion_mode == "add":
        print(f"[INFO] LLM admission threshold: {args.llm_threshold}")

    if args.mode.lower() == "scratch":
        if not args.pllm_file or not os.path.exists(args.pllm_file):
            raise FileNotFoundError(f"[ERROR] PLLM score file not found: {args.pllm_file}")
        print(f"[INFO] Loading pllm scores from: {args.pllm_file}")
        pllm_scores_map = load_pllm_scores_jsonl(args.pllm_file)
        print(f"[INFO] pllm loaded. num_sids={len(pllm_scores_map)}")
    else:
        pllm_scores_map = None
        print("[INFO] Guided mode: LLM fusion skipped.")

    all_type, relations = get_type_info(cfg.type_file)

    test_iter = get_data_loader2(
        cfg.test_prefix,
        cfg.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=cfg.max_sent_len,
        data_file=test_file
    )

    model = build_model(cfg, all_type)
    model = load_checkpoint(model, model_path, device)

    preds, elems, ents, sids, fusion_stats = infer_on_loader(
        model,
        test_iter,
        relations,
        all_type,
        device,
        pad_token_id,
        mode=args.mode,
        pllm_scores_map=pllm_scores_map,
        threshold=args.threshold,
        alpha=args.alpha,
        fusion_mode=fusion_mode,
        llm_threshold=args.llm_threshold,
        max_layer=args.max_layer,
    )

    flip_audit = None
    if args.audit_flips and args.mode.lower() == "scratch":
        print("[INFO] Flip audit: running alpha=1 baseline on the same test set.")
        baseline_preds, baseline_elems, baseline_ents, baseline_sids, _ = infer_on_loader(
            model,
            test_iter,
            relations,
            all_type,
            device,
            pad_token_id,
            mode=args.mode,
            pllm_scores_map=pllm_scores_map,
            threshold=args.threshold,
            alpha=1.0,
            fusion_mode="model",
            llm_threshold=args.llm_threshold,
            max_layer=args.max_layer,
        )
        if baseline_sids != sids or baseline_elems != elems or baseline_ents != ents:
            raise ValueError("alpha=1 baseline is not aligned with fused inference")
        flip_audit = audit_prediction_flips(
            sids, elems, ents, baseline_preds, preds
        )
        print_flip_audit(flip_audit)

    if args.debug_fusion:
        total_candidates = sum(s.get("candidate_count", 0) for s in fusion_stats)
        total_hits = sum(s.get("llm_hit_count", 0) for s in fusion_stats)
        total_structured = sum(s.get("llm_structured_hit_count", 0) for s in fusion_stats)
        total_legacy = sum(s.get("llm_legacy_hit_count", 0) for s in fusion_stats)
        total_high = sum(s.get("llm_ge_05_count", 0) for s in fusion_stats)
        total_admitted = sum(s.get("llm_admitted_count", 0) for s in fusion_stats)
        print(
            f"[DEBUG] Fusion candidate_count={total_candidates} | "
            f"llm_hit_count={total_hits} | structured_hit_count={total_structured} | "
            f"legacy_hit_count={total_legacy} | llm_ge_05_count={total_high} | "
            f"llm_admitted_count={total_admitted}"
        )
        for layer in sorted(set(s.get("layer") for s in fusion_stats)):
            layer_stats = [s for s in fusion_stats if s.get("layer") == layer]
            candidates = sum(s.get("candidate_count", 0) for s in layer_stats)
            hits = sum(s.get("llm_hit_count", 0) for s in layer_stats)
            structured = sum(s.get("llm_structured_hit_count", 0) for s in layer_stats)
            legacy = sum(s.get("llm_legacy_hit_count", 0) for s in layer_stats)
            high = sum(s.get("llm_ge_05_count", 0) for s in layer_stats)
            admitted = sum(s.get("llm_admitted_count", 0) for s in layer_stats)
            hit_rate = float(hits) / candidates if candidates else 0.0
            print(
                f"[DEBUG] Fusion L{layer} candidate_count={candidates} | "
                f"llm_hit_count={hits} | hit_rate={hit_rate:.6f} | "
                f"structured_hit_count={structured} | legacy_hit_count={legacy} | "
                f"llm_ge_05_count={high} | llm_admitted_count={admitted}"
            )

    p, r, f1 = compute_metric(elems, preds, ents)
    print(f"\n[RESULT] Precision={p} | Recall={r} | F1={f1}")

    if args.output_dir:
        out_dir = args.output_dir
    elif os.path.commonpath([os.path.abspath(model_path), os.path.abspath(cfg.model_dir)]) == os.path.abspath(cfg.model_dir):
        out_dir = cfg.model_dir
    else:
        out_dir = os.path.dirname(os.path.abspath(model_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "predict.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved predictions to: {out_path}")

    if flip_audit is not None:
        flip_path = os.path.join(out_dir, "fusion_flip_audit.json")
        with open(flip_path, "w", encoding="utf-8") as f:
            json.dump(flip_audit, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved flip audit to: {flip_path}")


if __name__ == "__main__":
    main()
