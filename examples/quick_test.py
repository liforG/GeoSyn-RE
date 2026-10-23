#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a dependency-free smoke test for the syntax candidate generator."""

from __future__ import print_function

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path, row):
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    # This row is already LTP-enriched, so the smoke test does not need LTP
    # model files. It exercises the same candidate-generation CLI used by the
    # full pipeline.
    row = {
        "sid": "quick-1",
        "sentence": "花岗岩是岩石。",
        "tokens": ["花岗岩", "是", "岩石", "。"],
        "pos": ["n", "v", "n", "wp"],
        "dep": [
            [1, 2, "SBV"],
            [2, 0, "HED"],
            [3, 2, "VOB"],
            [4, 2, "WP"],
        ],
        "entities": [
            {"text": "花岗岩", "type": "GS", "span": [0, 2]},
            {"text": "岩石", "type": "GS", "span": [4, 5]},
        ],
        "entity_map": {
            "0": [{
                "text": "花岗岩",
                "type": "GS",
                "span": [0, 2],
                "token_span": [0, 0],
            }],
            "2": [{
                "text": "岩石",
                "type": "GS",
                "span": [4, 5],
                "token_span": [2, 2],
            }],
        },
    }

    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_dir = Path(temporary_dir)
        input_path = temporary_dir / "input.jsonl"
        output_path = temporary_dir / "candidates.jsonl"
        audit_path = temporary_dir / "audit.jsonl"
        write_jsonl(input_path, row)

        command = [
            sys.executable,
            str(ROOT / "ltp_llm" / "generate_v29_candidates.py"),
            "--input",
            str(input_path),
            "--schema",
            str(ROOT / "data" / "type3.json"),
            "--output",
            str(output_path),
            "--audit-output",
            str(audit_path),
        ]
        subprocess.check_call(command, cwd=str(ROOT))

        with output_path.open("r", encoding="utf-8") as stream:
            result = json.loads(stream.readline())

    triples = result.get("triples", [])
    if len(triples) != 1 or triples[0].get("rid") != "EQ":
        raise AssertionError("expected one EQ candidate, got {!r}".format(triples))

    print("quick test passed: generated one EQ candidate")


if __name__ == "__main__":
    main()
