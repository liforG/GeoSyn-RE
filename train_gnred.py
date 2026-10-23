#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the supervised GeoSyn-RE model on the bundled GNRED split."""

from __future__ import print_function

import argparse
import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))


def rooted(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def require_file(path):
    if not os.path.isfile(path):
        raise IOError("missing required file: {}".format(path))


def main():
    parser = argparse.ArgumentParser(description="Train GeoSyn-RE on GNRED")
    parser.add_argument("--data-dir", default="data/gnred")
    parser.add_argument("--type-file", default="data/type3.json")
    parser.add_argument("--bert-path", default="bert_geo")
    parser.add_argument("--output-dir", default="checkpoints/gnred_seed66")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--standard-only", action="store_true")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(rooted(args.data_dir))
    args.type_file = os.path.abspath(rooted(args.type_file))
    args.bert_path = os.path.abspath(rooted(args.bert_path))
    args.output_dir = os.path.abspath(rooted(args.output_dir))

    if args.batch_size < 1 or args.epochs < 1 or args.early_stop < 1:
        raise ValueError("batch size, epochs, and early stop must all be positive")
    for split in ("train", "valid", "test"):
        require_file(os.path.join(args.data_dir, split + ".json"))
    require_file(args.type_file)
    for name in ("config.json", "vocab.txt"):
        require_file(os.path.join(args.bert_path, name))
    if not any(os.path.isfile(os.path.join(args.bert_path, name))
               for name in ("pytorch_model.bin", "model.safetensors")):
        raise IOError("BERT directory has no model weights")

    with open(os.path.join(args.bert_path, "config.json"), "r", encoding="utf-8") as stream:
        hidden_size = int(json.load(stream)["hidden_size"])
    if hidden_size % 4:
        raise ValueError("BERT hidden_size must be divisible by four attention heads")

    print("[OK] GNRED inputs are ready")
    print("[INFO] data={}".format(args.data_dir))
    print("[INFO] schema={}".format(args.type_file))
    print("[INFO] encoder={}".format(args.bert_path))
    print("[INFO] output={}".format(args.output_dir))
    from config import Config
    import data_utils
    from framework import FrameWorkTrans
    from utils import set_random

    config = Config()
    config.data_dir = args.data_dir
    config.type_file = args.type_file
    config.radical_vocab_file = os.path.join(args.data_dir, "train.json")
    config.bert_dir = args.bert_path
    config.bert_hidden = hidden_size
    config.model_dir = args.output_dir
    config.device = args.device
    config.random_seed = args.seed
    config.batch_size = args.batch_size
    config.train_epoch = args.epochs
    config.early_stop = args.early_stop
    config.bert_lr = 1e-5
    config.lr = 2e-5
    config.cont_weight = 0
    config.trans_type = 2
    config.predict_threshold = args.threshold
    config.max_relation_layer = args.max_layer
    config.guided_max_layer = args.max_layer
    config.guided_compute_auc = False
    config.use_schema = True
    config.guide = not args.standard_only

    data_utils.config.data_dir = config.data_dir
    data_utils.config.type_file = config.type_file
    data_utils.config.radical_vocab_file = config.radical_vocab_file
    data_utils.config.max_sent_len = config.max_sent_len
    data_utils.config.use_schema = config.use_schema

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise IOError("output directory is not empty: {}".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "invocation.json"), "w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, ensure_ascii=False, indent=2, sort_keys=True)

    set_random(config.random_seed)
    FrameWorkTrans(config).train()
    return 0


if __name__ == "__main__":
    main()
