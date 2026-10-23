#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified GNRED entry for training and the V29 LTP + LLM API pipeline."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def rooted(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def require_file(path, label):
    if not os.path.isfile(path):
        raise IOError("missing {}: {}".format(label, path))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run(command):
    print("[RUN] {}".format(" ".join(command)))
    subprocess.check_call(command, cwd=ROOT)


def write_manifest(path, args, files):
    payload = {
        "arguments": vars(args),
        "python": sys.version,
        "files": {},
    }
    for label, filename in sorted(files.items()):
        payload["files"][label] = {
            "path": os.path.abspath(filename),
            "sha256": sha256_file(filename) if os.path.isfile(filename) else None,
        }
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)


def add_shared_paths(parser):
    parser.add_argument("--data-dir", default="data/gnred")
    parser.add_argument("--type-file", default="data/type3.json")
    parser.add_argument("--bert-path", default="bert_geo")


def build_parser():
    parser = argparse.ArgumentParser(description="GeoSyn-RE reproducibility entry")
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="train the supervised GNRED model")
    add_shared_paths(train)
    train.add_argument("--output-dir", default="checkpoints/gnred_seed66")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, default=66)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--early-stop", type=int, default=20)
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--standard-only", action="store_true")

    pipeline = subparsers.add_parser(
        "pipeline", help="run LTP V29, LLM API scoring, and supervised fusion"
    )
    add_shared_paths(pipeline)
    pipeline.add_argument("--checkpoint", default="checkpoints/gnred_seed66/model.pt")
    pipeline.add_argument("--output-dir", default="outputs/gnred_v29_api")
    pipeline.add_argument("--device", default="cuda:0")
    pipeline.add_argument("--threshold", type=float, default=0.60)
    pipeline.add_argument("--batch-size", type=int, default=1)
    pipeline.add_argument("--pos-model", default="assets/ltp/pos.model")
    pipeline.add_argument("--parser-model", default="assets/ltp/parser.model")
    pipeline.add_argument("--api-provider", choices=("deepseek", "openai"), default="deepseek")
    pipeline.add_argument("--api-base-url", default=None,
                          help="API root; defaults from --api-provider")
    pipeline.add_argument("--api-model", default=None,
                          help="model name; defaults from --api-provider")
    pipeline.add_argument("--api-key-env", default=None,
                          help="key environment variable; defaults from --api-provider")
    pipeline.add_argument("--api-batch-size", type=int, default=30)
    pipeline.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    pipeline.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    pipeline.add_argument("--max-tokens", type=int, default=16384)
    pipeline.add_argument("--request-timeout", type=int, default=300)
    pipeline.add_argument("--max-retries", type=int, default=3)
    pipeline.add_argument("--retry-sleep", type=float, default=2.0)
    pipeline.add_argument("--resume-api", action="store_true")
    pipeline.add_argument("--overwrite-api", action="store_true")
    pipeline.add_argument("--dry-run", action="store_true",
                          help="prepare all inputs and validate the API plan without calling it")
    pipeline.add_argument(
        "--stop-after",
        choices=("ltp", "candidates", "tasks", "api", "fusion", "compare"),
        default="compare",
        help="stop after a stage for progressive verification",
    )
    return parser


def train(args):
    command = [
        PYTHON, os.path.join(ROOT, "train_gnred.py"),
        "--data-dir", rooted(args.data_dir),
        "--type-file", rooted(args.type_file),
        "--bert-path", rooted(args.bert_path),
        "--output-dir", rooted(args.output_dir),
        "--device", args.device,
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--early-stop", str(args.early_stop),
        "--threshold", str(args.threshold),
        "--max-layer", "3",
    ]
    if args.standard_only:
        command.append("--standard-only")
    run(command)


def pipeline(args):
    data_dir = os.path.abspath(rooted(args.data_dir))
    test_file = os.path.join(data_dir, "test.json")
    train_file = os.path.join(data_dir, "train.json")
    type_file = os.path.abspath(rooted(args.type_file))
    bert_path = os.path.abspath(rooted(args.bert_path))
    checkpoint = os.path.abspath(rooted(args.checkpoint))
    pos_model = os.path.abspath(rooted(args.pos_model))
    parser_model = os.path.abspath(rooted(args.parser_model))
    output_dir = os.path.abspath(rooted(args.output_dir))

    for path, label in (
        (test_file, "GNRED test split"),
        (train_file, "GNRED train split"),
        (type_file, "GNRED schema"),
        (pos_model, "LTP POS model"),
        (parser_model, "LTP parser model"),
    ):
        require_file(path, label)
    if args.resume_api and args.overwrite_api:
        raise ValueError("--resume-api and --overwrite-api are mutually exclusive")

    ltp_dir = os.path.join(output_dir, "ltp")
    candidate_dir = os.path.join(output_dir, "candidates")
    llm_dir = os.path.join(output_dir, "llm")
    fusion_dir = os.path.join(output_dir, "fusion")
    compare_dir = os.path.join(output_dir, "compare")
    for directory in (output_dir, ltp_dir, candidate_dir, llm_dir, fusion_dir, compare_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    enriched = os.path.join(ltp_dir, "test_enriched.jsonl")
    candidates = os.path.join(candidate_dir, "test_triples_ids.jsonl")
    evidence_audit = os.path.join(candidate_dir, "test_evidence_audit.jsonl")
    tasks = os.path.join(llm_dir, "tasks.jsonl")
    scored = os.path.join(llm_dir, "triples_scored.jsonl")
    raw = os.path.join(llm_dir, "triples_scored.raw.jsonl")
    api_metadata = os.path.join(llm_dir, "triples_scored.run.json")
    pllm = os.path.join(fusion_dir, "pllm_scores.jsonl")
    missing = os.path.join(fusion_dir, "missing.json")

    manifest_files = {
        "test": test_file,
        "train": train_file,
        "schema": type_file,
        "bert_config": os.path.join(bert_path, "config.json"),
        "ltp_pos": pos_model,
        "ltp_parser": parser_model,
        "checkpoint": checkpoint,
    }
    write_manifest(os.path.join(output_dir, "run_manifest.json"), args, manifest_files)

    run([
        PYTHON, os.path.join(ROOT, "ltp_llm", "run_batch_2.py"),
        "--input", test_file,
        "--out-dir", ltp_dir,
        "--output", enriched,
        "--pos-model", pos_model,
        "--parser-model", parser_model,
    ])
    if args.stop_after == "ltp":
        return

    run([
        PYTHON, os.path.join(ROOT, "ltp_llm", "generate_v29_candidates.py"),
        "--input", enriched,
        "--schema", type_file,
        "--output", candidates,
        "--audit-output", evidence_audit,
    ])
    if args.stop_after == "candidates":
        return

    run([
        PYTHON, os.path.join(ROOT, "ltp_llm", "build_llm_candidate_tasks.py"),
        "--input", candidates,
        "--enriched", enriched,
        "--type-file", type_file,
        "--output", tasks,
    ])
    if args.stop_after == "tasks":
        return

    api_command = [
        PYTHON, os.path.join(ROOT, "ltp_llm", "llm_api_client.py"),
        "--input", candidates,
        "--tasks", tasks,
        "--output", scored,
        "--raw-output", raw,
        "--run-metadata", api_metadata,
        "--dataset-role", "test",
        "--prompt-module", "llm_candidate_prompt",
        "--hierarchy-mode", "raw",
        "--provider", args.api_provider,
        "--batch-size", str(args.api_batch_size),
        "--thinking", args.thinking,
        "--reasoning-effort", args.reasoning_effort,
        "--temperature", "0.0",
        "--top-p", "1.0",
        "--max-tokens", str(args.max_tokens),
        "--timeout", str(args.request_timeout),
        "--max-retries", str(args.max_retries),
        "--retry-sleep", str(args.retry_sleep),
    ]
    if args.api_base_url:
        api_command.extend(["--base-url", args.api_base_url])
    if args.api_model:
        api_command.extend(["--model", args.api_model])
    if args.api_key_env:
        api_command.extend(["--api-key-env", args.api_key_env])
    if args.resume_api:
        api_command.append("--resume")
    if args.overwrite_api:
        api_command.append("--overwrite")
    if args.dry_run:
        api_command.append("--dry-run")
    run(api_command)
    if args.dry_run or args.stop_after == "api":
        return

    run([
        PYTHON, os.path.join(ROOT, "ltp_llm", "for_element_score2.py"),
        "--test", test_file,
        "--triples", scored,
        "--output", pllm,
        "--missing-output", missing,
    ])
    if args.stop_after == "fusion":
        return

    require_file(checkpoint, "supervised checkpoint")
    run([
        PYTHON, os.path.join(ROOT, "inference_llm_compare.py"),
        "--model", checkpoint,
        "--test", test_file,
        "--data-dir", data_dir,
        "--type-file", type_file,
        "--radical-vocab-file", train_file,
        "--bert-path", bert_path,
        "--device", args.device,
        "--max-sent-len", "128",
        "--max-layer", "3",
        "--pllm-file", pllm,
        "--threshold", str(args.threshold),
        "--batch-size", str(args.batch_size),
        "--output-dir", compare_dir,
        "--audit-flips",
    ])


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.error("a command is required: train or pipeline")
    if args.command == "train":
        train(args)
    else:
        pipeline(args)


if __name__ == "__main__":
    main()
