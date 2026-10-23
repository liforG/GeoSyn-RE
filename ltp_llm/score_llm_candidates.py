#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Score sentence-grouped candidates with a supported LLM API.

The runner is intentionally restart-safe: every successful batch is validated
and appended to a checkpoint JSONL before the next paid request starts.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib
import json
import math
import os
import time

import requests

from build_llm_candidate_tasks import (
    iter_group_batches,
    read_jsonl,
    validate_candidate_groups,
)
from llm_hierarchy_scores import apply_final_scores, compute_hierarchy_scores


def write_jsonl_atomic(path, rows):
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, absolute)


def write_json_atomic(path, value):
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, absolute)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def batch_task_ids(batch):
    return [
        str(candidate["task_id"])
        for group in batch
        for candidate in group.get("candidates", [])
    ]


PROVIDER_DEFAULTS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.6",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def resolve_provider_args(args):
    defaults = PROVIDER_DEFAULTS[args.provider]
    if args.base_url is None:
        args.base_url = defaults["base_url"]
    if args.model is None:
        args.model = defaults["model"]
    if args.api_key_env is None:
        args.api_key_env = defaults["api_key_env"]
    if args.provider == "deepseek" and args.thinking == "enabled":
        if args.reasoning_effort not in ("high", "max"):
            raise ValueError(
                "DeepSeek thinking mode supports reasoning-effort high or max in this runner"
            )


def effective_reasoning_effort(args):
    if args.provider == "openai" and args.thinking == "disabled":
        return "none"
    if args.thinking == "enabled":
        return args.reasoning_effort
    return None


def request_config(args, prompt_version):
    return {
        "provider": args.provider,
        "api_protocol": "chat_completions",
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "prompt_version": prompt_version,
        "prompt_module": args.prompt_module,
        "message_layout": "single_user_full_prompt",
        "thinking": args.thinking,
        "reasoning_effort": effective_reasoning_effort(args),
        "temperature": args.temperature if args.provider == "deepseek" else None,
        "top_p": args.top_p if args.provider == "deepseek" else None,
        "max_tokens": args.max_tokens,
        "response_format_json": bool(args.response_format_json),
        "batch_size": args.batch_size,
        "hierarchy_mode": args.hierarchy_mode,
        "network_tools": [],
    }


def validate_batch_predictions(predictions, expected_task_ids, batch_no):
    if not isinstance(predictions, list):
        raise ValueError("batch {} response.results is not a JSON array".format(batch_no))

    parsed = {}
    duplicate = []
    bad_items = []
    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            bad_items.append(index)
            continue
        task_id = str(item.get("task_id", "")).strip()
        confidence = item.get("confidence")
        if not task_id or isinstance(confidence, bool):
            bad_items.append(index)
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            bad_items.append(index)
            continue
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            bad_items.append(index)
            continue
        if task_id in parsed:
            duplicate.append(task_id)
        parsed[task_id] = confidence

    expected = set(expected_task_ids)
    actual = set(parsed)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if bad_items or duplicate or missing or extra or len(predictions) != len(expected_task_ids):
        raise ValueError(
            "batch {} invalid predictions: items={} expected={} bad={} duplicate={} missing={} extra={}".format(
                batch_no,
                len(predictions),
                len(expected_task_ids),
                bad_items[:10],
                sorted(set(duplicate))[:10],
                missing[:10],
                extra[:10],
            )
        )

    # Preserve task input order even if the API returns a different order.
    return [
        {"task_id": task_id, "confidence": parsed[task_id]}
        for task_id in expected_task_ids
    ]


def add_usage(usage_totals, usage):
    usage = usage or {}
    for key in usage_totals:
        usage_totals[key] += int(usage.get(key, 0) or 0)


def build_request_payload(args, prompt):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if args.provider == "deepseek":
        payload.update({
            "thinking": {"type": args.thinking},
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        })
        if args.thinking == "enabled":
            payload["reasoning_effort"] = args.reasoning_effort
    else:
        # GPT reasoning models count visible output and reasoning tokens against
        # max_completion_tokens. DeepSeek's `thinking` field must not be sent.
        payload["max_completion_tokens"] = args.max_tokens
        payload["reasoning_effort"] = effective_reasoning_effort(args)
    if args.response_format_json:
        payload["response_format"] = {"type": "json_object"}
    return payload


def call_llm_api(
    args,
    api_key,
    sentence_groups,
    prompt_builder,
    result_extractor,
):
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }
    payload = build_request_payload(args, prompt_builder(sentence_groups))

    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=args.timeout)
            if response.status_code >= 400:
                raise RuntimeError("HTTP {}: {}".format(response.status_code, response.text[:1000]))
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content") or ""
            if finish_reason == "length":
                raise RuntimeError("response truncated by max_tokens; content prefix: {}".format(content[:1000]))
            if not content.strip():
                raise RuntimeError("response content is empty")
            predictions = result_extractor(content)
            response_meta = {
                "response_id": data.get("id"),
                "response_model": data.get("model"),
                "finish_reason": finish_reason,
                "usage": data.get("usage", {}),
                "reasoning_content": reasoning_content,
            }
            return content, predictions, response_meta
        except Exception as exc:
            last_error = exc
            print("[WARN] API attempt {}/{} failed: {}".format(attempt, args.max_retries, exc))
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * attempt)
    raise RuntimeError("{} API call failed after {} retries: {}".format(
        args.provider, args.max_retries, last_error
    ))


def load_checkpoints(path, run_signature, batches):
    if not os.path.exists(path):
        return {}
    expected_by_batch = {
        index: batch_task_ids(batch)
        for index, batch in enumerate(batches, 1)
    }
    checkpoints = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError("{}:{} invalid checkpoint JSON: {}".format(path, line_no, exc))
            batch_no = int(record.get("batch", 0))
            if batch_no not in expected_by_batch:
                raise ValueError("checkpoint has unknown batch {}".format(batch_no))
            if batch_no in checkpoints:
                raise ValueError("checkpoint has duplicate batch {}".format(batch_no))
            if record.get("run_signature") != run_signature:
                raise ValueError("checkpoint batch {} belongs to a different run configuration".format(batch_no))
            if [str(value) for value in record.get("tasks", [])] != expected_by_batch[batch_no]:
                raise ValueError("checkpoint batch {} task list does not match current tasks".format(batch_no))
            record["parsed"] = validate_batch_predictions(
                record.get("parsed"), expected_by_batch[batch_no], batch_no
            )
            checkpoints[batch_no] = record
    return checkpoints


def strict_hierarchy_scores(groups, predictions, hierarchy_mode):
    if hierarchy_mode == "min":
        final_by_task, summary = compute_hierarchy_scores(groups, predictions)
        problems = {
            key: summary[key]
            for key in ("missing_tasks", "extra_tasks", "duplicate_tasks", "bad_items")
            if summary[key]
        }
        if problems:
            raise ValueError("invalid complete prediction set: {}".format(problems))
        return final_by_task, summary

    expected_task_ids = batch_task_ids(groups)
    validated = validate_batch_predictions(
        predictions, expected_task_ids, "complete"
    )
    return (
        {
            item["task_id"]: float(item["confidence"])
            for item in validated
        },
        {
            "hierarchy_adjusted": 0,
            "hierarchy_adjusted_by_layer": {},
            "missing_tasks": [],
            "extra_tasks": [],
            "duplicate_tasks": [],
            "bad_items": [],
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Restart-safe DeepSeek/OpenAI scoring for candidate graphs."
    )
    parser.add_argument("--input", required=True, help="Input triples_ids JSONL path.")
    parser.add_argument("--tasks", required=True, help="Sentence-grouped task JSONL.")
    parser.add_argument("--output", required=True, help="Output triples_scored JSONL path.")
    parser.add_argument("--raw-output", default=None,
                        help="Batch checkpoint JSONL; defaults to <output>.raw.jsonl.")
    parser.add_argument("--run-metadata", default=None,
                        help="Run metadata JSON; defaults to <output>.run.json.")
    parser.add_argument("--dataset-role", choices=("validation", "test", "custom"), default="custom")
    parser.add_argument(
        "--prompt-module",
        default="llm_candidate_prompt",
        help=(
            "Importable module exposing PROMPT_VERSION, build_full_prompt "
            "and extract_results."
        ),
    )
    parser.add_argument(
        "--hierarchy-mode",
        choices=("min", "raw"),
        default="min",
        help=(
            "Use legacy child-min propagation or retain the LLM score for "
            "each complete candidate tree. Formal compare experiments use raw."
        ),
    )
    parser.add_argument("--provider", choices=("deepseek", "openai"), default="deepseek")
    parser.add_argument("--base-url", default=None,
                        help="Provider API root; defaults from --provider.")
    parser.add_argument("--model", default=None,
                        help="Model name; defaults from --provider.")
    parser.add_argument("--api-key-env", default=None,
                        help="Environment variable containing the key; defaults from --provider.")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Approximate candidates per request; sentence groups are never split.")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--no-response-format-json", dest="response_format_json", action="store_false")
    parser.set_defaults(response_format_json=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse validated batches from an existing raw checkpoint.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Start a new run and overwrite existing checkpoint/output files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print the run plan without reading an API key or calling the API.")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="API smoke test; checkpoints batches but does not write a partial scored output.")
    args = parser.parse_args()

    resolve_provider_args(args)

    if args.batch_size <= 0 or args.max_retries <= 0 or args.max_tokens <= 0:
        raise ValueError("batch-size, max-retries and max-tokens must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    prompt_module = importlib.import_module(args.prompt_module)
    prompt_version = str(prompt_module.PROMPT_VERSION)
    prompt_builder = prompt_module.build_full_prompt
    result_extractor = prompt_module.extract_results

    raw_output = args.raw_output or (args.output + ".raw.jsonl")
    run_metadata = args.run_metadata or (args.output + ".run.json")
    rows = read_jsonl(args.input)
    groups = read_jsonl(args.tasks)
    task_summary = validate_candidate_groups(groups, source_rows=rows)
    batches = list(iter_group_batches(groups, args.batch_size))
    selected_batches = batches
    if args.limit_batches is not None:
        if args.limit_batches <= 0:
            raise ValueError("limit-batches must be positive")
        selected_batches = batches[:args.limit_batches]

    config = request_config(args, prompt_version)
    signature_payload = {
        "request": config,
        "input_sha256": sha256_file(args.input),
        "tasks_sha256": sha256_file(args.tasks),
    }
    run_signature = stable_hash(signature_payload)
    metadata = {
        "status": "dry_run" if args.dry_run else "running",
        "dataset_role": args.dataset_role,
        "input": os.path.abspath(args.input),
        "tasks": os.path.abspath(args.tasks),
        "output": os.path.abspath(args.output),
        "raw_output": os.path.abspath(raw_output),
        "run_signature": run_signature,
        "signature_payload": signature_payload,
        "task_summary": task_summary,
        "total_batches": len(batches),
        "selected_batches": len(selected_batches),
        "selected_candidates": sum(len(batch_task_ids(batch)) for batch in selected_batches),
    }

    print("[PLAN] dataset_role:", args.dataset_role)
    print("[PLAN] prompt_version:", prompt_version)
    print("[PLAN] hierarchy_mode:", args.hierarchy_mode)
    print("[PLAN] provider:", args.provider)
    print("[PLAN] model:", args.model)
    print("[PLAN] thinking:", args.thinking)
    print("[PLAN] reasoning_effort:", config["reasoning_effort"])
    print("[PLAN] network_tools: []")
    print("[PLAN] rows:", len(rows))
    print("[PLAN] sentence_groups:", task_summary["groups"])
    print("[PLAN] unique_candidates:", task_summary["candidates"])
    print("[PLAN] source_candidates:", task_summary["source_candidates"])
    print("[PLAN] batches:", len(selected_batches), "/", len(batches))
    print("[PLAN] run_signature:", run_signature)

    if args.dry_run:
        write_json_atomic(run_metadata, metadata)
        print("[DRY-RUN] no API key read and no API request sent")
        print("[DRY-RUN] metadata:", run_metadata)
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError("Missing API key. Please set environment variable {}".format(args.api_key_env))

    if args.overwrite:
        for path in (raw_output, args.output, run_metadata):
            if os.path.exists(path):
                os.remove(path)
    elif os.path.exists(raw_output) and not args.resume:
        raise RuntimeError("checkpoint already exists; use --resume or --overwrite: {}".format(raw_output))

    checkpoints = load_checkpoints(raw_output, run_signature, batches) if args.resume else {}
    metadata["resumed_batches"] = sorted(checkpoints)
    write_json_atomic(run_metadata, metadata)
    os.makedirs(os.path.dirname(os.path.abspath(raw_output)), exist_ok=True)

    all_predictions = []
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    mode = "a" if args.resume and os.path.exists(raw_output) else "w"
    with open(raw_output, mode, encoding="utf-8") as raw_stream:
        for batch_no, batch in enumerate(selected_batches, 1):
            expected_task_ids = batch_task_ids(batch)
            if batch_no in checkpoints:
                predictions = checkpoints[batch_no]["parsed"]
                all_predictions.extend(predictions)
                add_usage(
                    usage_totals,
                    checkpoints[batch_no].get("response", {}).get("usage", {}),
                )
                print("[RESUME] batch {} | candidates={}".format(batch_no, len(predictions)))
                continue

            print("[INFO] scoring batch {} | sentence_groups={} | candidates={}".format(
                batch_no, len(batch), len(expected_task_ids)
            ))
            raw_text, predictions, response_meta = call_llm_api(
                args,
                api_key,
                batch,
                prompt_builder,
                result_extractor,
            )
            predictions = validate_batch_predictions(predictions, expected_task_ids, batch_no)
            record = {
                "batch": batch_no,
                "run_signature": run_signature,
                "prompt_version": prompt_version,
                "group_ids": [group["group_id"] for group in batch],
                "tasks": expected_task_ids,
                "raw_response": raw_text,
                "parsed": predictions,
                "response": response_meta,
            }
            raw_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
            all_predictions.extend(predictions)
            add_usage(usage_totals, response_meta.get("usage", {}))

    if len(selected_batches) != len(batches):
        metadata["status"] = "partial_checkpoint_complete"
        metadata["usage"] = usage_totals
        write_json_atomic(run_metadata, metadata)
        print("[PARTIAL] checkpointed {} batches; no partial scored output written".format(
            len(selected_batches)
        ))
        return

    final_by_task, hierarchy_summary = strict_hierarchy_scores(
        groups, all_predictions, args.hierarchy_mode
    )
    total, missing = apply_final_scores(rows, groups, final_by_task)
    if missing:
        raise ValueError("source candidates missing final scores: {}".format(missing[:10]))
    write_jsonl_atomic(args.output, rows)

    metadata["status"] = "complete"
    metadata["usage"] = usage_totals
    metadata["prediction_items"] = len(all_predictions)
    metadata["scored_source_candidates"] = total
    metadata["hierarchy_summary"] = hierarchy_summary
    write_json_atomic(run_metadata, metadata)

    print("[DONE] rows:", len(rows))
    print("[DONE] sentence_groups:", task_summary["groups"])
    print("[DONE] unique_candidates:", task_summary["candidates"])
    print("[DONE] source_candidates:", total)
    print("[DONE] predictions:", len(all_predictions))
    print("[DONE] hierarchy_adjusted:", hierarchy_summary["hierarchy_adjusted"])
    print("[DONE] hierarchy_adjusted_by_layer:", json.dumps(
        hierarchy_summary["hierarchy_adjusted_by_layer"], ensure_ascii=False, sort_keys=True
    ))
    print("[DONE] usage:", json.dumps(usage_totals, ensure_ascii=False, sort_keys=True))
    print("[DONE] output:", args.output)
    print("[DONE] raw_output:", raw_output)
    print("[DONE] run_metadata:", run_metadata)


if __name__ == "__main__":
    main()
