# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Site    :
# @File    : models.py
# @Software: PyCharm

import sys
import hashlib
import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, AutoModel, RobertaModel
import numpy as np
import copy
from tiny_models import TreeLSTM, Attention, LSTMEncoder, do_pad, Classifier
from tiny_models import MyPositionalEncoding, MyTransformerEncoder
from data_utils import has_candidate, generate_re_type2id, generate_type2index, generate_candidate1
from data_utils import generate_candidate2, generate_candidate3
from utils import compute_similarity_examples
from radicalview import char2radical, construct_radical_vocab_from_vocab, alphabet2vocab, RadicalViewFeature
from alphabet import Alphabet
from build import build_alphabet
from config import Config
import copy


def _top_model_candidates_by_operand(candidates, scores, threshold, eligible=None):
    """Return the model's best above-threshold relation for each operand pair."""
    grouped = {}
    for index, candidate in enumerate(candidates):
        if eligible is not None and not eligible[index]:
            continue
        score = float(scores[index].item())
        if score <= threshold:
            continue
        operand_pair = candidate[1]
        previous = grouped.get(operand_pair)
        if previous is None or score > previous[1]:
            grouped[operand_pair] = (index, score)
    return {value[0] for value in grouped.values()}


def _select_scratch_candidates(
    candidates,
    model_scores,
    final_scores,
    threshold,
    fusion_mode,
    llm_admitted=None,
    model_eligible=None,
):
    """Select runtime candidates while keeping add mode strictly additive."""
    if fusion_mode != "add":
        return _top_model_candidates_by_operand(
            candidates, final_scores, threshold, eligible=model_eligible
        )

    selected = _top_model_candidates_by_operand(
        candidates, model_scores, threshold, eligible=model_eligible
    )
    if llm_admitted is not None:
        selected.update(
            index
            for index in range(len(candidates))
            if bool(llm_admitted[index].item())
        )
    return selected




class ITransformer(nn.Module):
    def __init__(self, bert_path, max_seq_len, hidden_size, head, trans_layers, all_types,
                 trans_type=1, runtime_config=None):  # 实例化时传来的
        super(ITransformer, self).__init__()
        self.encoder = BertModel.from_pretrained(bert_path)
        self.DAG_trans = MyTransformerEncoder(hidden_size, 0.1, head, 4 * hidden_size, trans_layers)
        # self.pos_encoder = PositionalEncoding(hidden_size, max_seq_len)
        self.pos_encoder = MyPositionalEncoding(hidden_size)
        self.type2id = {tp: i for i, tp in enumerate(all_types)}
        self.type_embedding = nn.Embedding(len(all_types) + 2,
                                           hidden_size)
        self.type_trans = nn.Linear(hidden_size, hidden_size)
        self.hid_trans = nn.Linear(hidden_size, hidden_size)
        self.cls_id, self.sep_id = len(all_types), len(all_types) + 1
        self.attention = Attention(hidden_size)
        self.cls = Classifier(in_features=hidden_size)
        self.trans_type = trans_type
        config = runtime_config if runtime_config is not None else Config()

        input_file = getattr(config, "radical_vocab_file", None)
        if not input_file:
            input_file = os.path.join(config.data_dir, "train.json")
        if not os.path.exists(input_file):
            raise FileNotFoundError("radical vocabulary source not found: {}".format(input_file))
        word_alphabet = Alphabet("word")
        word_alphabet, train_freq, biword_alphabet, char_alphabet = build_alphabet(input_file, word_alphabet)
        filter_nums = config.filter_nums
        kernel_sizes = config.kernel_sizes
        pool_method = config.pool_method
        self.radical_encoder = RadicalViewFeature(config=config, word_alphabet=word_alphabet, filter_nums=filter_nums,
                                                  kernel_sizes=kernel_sizes, pool_method=pool_method,
                                                  include_word_start_end=True).to(
            config.device)
        self.radical_vocab_metadata = self._build_radical_vocab_metadata(input_file)

    @staticmethod
    def _ordered_vocab(vocab):
        return [word for word, _ in sorted(list(vocab), key=lambda item: item[1])]

    def _build_radical_vocab_metadata(self, source_file):
        digest = hashlib.sha256()
        with open(source_file, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "source_file": os.path.normpath(source_file),
            "source_sha256": digest.hexdigest(),
            "chars": self._ordered_vocab(self.radical_encoder.chars_vocab),
            "radicals": self._ordered_vocab(self.radical_encoder.radical_vocab),
        }

    def get_radical_vocab_metadata(self):
        return copy.deepcopy(self.radical_vocab_metadata)

    def validate_radical_vocab_metadata(self, metadata):
        if not metadata:
            print("[WARN] legacy checkpoint has no radical vocabulary metadata")
            return
        current = self.radical_vocab_metadata
        for key in ("chars", "radicals"):
            if metadata.get(key) != current.get(key):
                raise RuntimeError(
                    "radical vocabulary mismatch for {}: checkpoint={} current={}".format(
                        key, len(metadata.get(key, [])), len(current.get(key, []))
                    )
                )

    def forward(self, x, x_mask, entity_index, entity_type, elements, relations, sim_label=None, c_w=0):
        """
        train model iteratively layer by layer
        :param x: input words ids b x s  x 表示的是输入的单词 ID，形状为 [b, s]，其中： b 是批量大小（batch size）。 s 是序列长度（sequence length） ids代表了句子sentence
        :param x_mask: input mask b x s
        :param entity_index: b x ?s x 2, start and end index of all entity in sentences of a batch  实体起始位置
        :param entity_type: b x ?s, entity types in sentences of a batch
        :param elements: b x ?n, element of all sentences in a batch, [(type: str, (op_idx, ...))]
        :param relations: r, contain type and ops types (type: str, (op_type))
        :return:
        """

        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[
            0]  # b x l  x h
        radical_encoder_output = self.radical_encoder(x)  # x
        hs = encoder_output.shape[-1]
        loss_c = t.tensor(0)
        if c_w != 0:
            cls_hidden = encoder_output[:, 0]  # b x h
            sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
            sim_mat = t.triu(sim_mat, diagonal=1)
            sim_mat = t.exp(sim_mat) / (1 + t.exp(sim_mat))
            sim_label = t.tensor(sim_label, dtype=t.float, device=device)
            loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)
        radical_hidden = [
            radical_encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x    ?s x mw x h
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h

        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in
                       range(b)]
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden,
                                           entity_mask)]
        all_radical_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                              if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                              for h, m in zip(radical_hidden, entity_mask)]

        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        loss, num, layer = None, 0, 1
        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
        #               for i in range(b)]
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        while has_candidate(candidates):
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_label = [[c in elements[i] for c in candidates[i]] + [False] * (max_c_len - c_lens[i])
                               for i in range(b)]
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)

            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])

                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, op1_rad, sep_emb, op2_emb, op2_hid,
                     op2_rad],
                    dim=-1).reshape(
                    [b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])

                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, op1_rad, sep_emb, op2_emb, op2_hid,
                     op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, op1_rad, sep_emb, op2_hid, op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, op1_rad, sep_emb, op2_hid, op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:,
                        0]
            trans_out = trans_out.reshape(
                [b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc

            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False, )
            t_loss = t.sum(t_loss * candidate_mask)
            loss = t_loss if not loss else (t_loss + loss)
            num += candidate_mask.sum().item()

            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(c_lens[i]):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        all_radical_hidden[i][idx] = trans_out[i][j]
                        type2index[i][candidates[i][j][0]][-1].append(idx)

            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        return (loss / num if loss else t.tensor(0)) * (1 - c_w) + loss_c * c_w

    def forward_cont(self, x, x_mask, entity_index, entity_type, elements, relations, sim_label, c_w=0.1):
        b, device = x.shape[0], x.device
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        cls_hidden = encoder_output[:, 0]  # b x h
        sim_mat = t.matmul(cls_hidden, cls_hidden.t()).clamp(-9, 9)
        sim_mat = t.triu(sim_mat, diagonal=1)
        sim_mat = t.exp(sim_mat) / (1 + t.exp(sim_mat))
        sim_label = t.tensor(sim_label, dtype=t.float, device=device)
        loss_c = F.smooth_l1_loss(sim_mat, sim_label, size_average=True)

        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h


        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        loss, num, layer = None, 0, 1
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        while has_candidate(candidates):
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  #
            candidate_label = [[c in elements[i] for c in candidates[i]] + [False] * (max_c_len - c_lens[i])
                               for i in range(b)]
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)

            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc

            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False, )
            t_loss = t.sum(t_loss * candidate_mask)
            loss = t_loss if not loss else (t_loss + loss)
            num += candidate_mask.sum().item()

            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(c_lens[i]):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        # type2index[i][candidates[i][j][0]].append(idx)
                        type2index[i][candidates[i][j][0]][-1].append(idx)
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        return (loss / num if loss else t.tensor(0)) * (1 - c_w) + loss_c * c_w

    def predict_scratch(self, x, x_mask, entity_index, entity_type, relations, threshold=0.5,
                            pllm_scores=None, alpha=0.7, fusion_mode="weighted",
                            llm_threshold=0.95, max_layer=6):
        """
        predict = self.model.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations)
        predict result from provided entities layer by layer
        """
        if fusion_mode not in ("weighted", "add", "model", "compare"):
            raise ValueError("unsupported LLM fusion mode: {}".format(fusion_mode))
        self.last_llm_fusion_stats = []
        b, device = x.shape[0], x.device

        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_ew_len = max([max(s) if s else 0 for s in entity_words_len])
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]
        radical_encoder_output = self.radical_encoder(x)  # x
        hs = encoder_output.shape[-1]
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])  # 填充0 使得长度相同
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h

        radical_hidden = [  #
            radical_encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]

        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=h.shape[0] + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([h.shape[0] + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]
        all_radical_hidden = [do_pad(self.attention(h, m), max_l=h.shape[0] + 1, pad_elem=0)
                              if h.shape[0] else t.zeros([h.shape[0] + 1, hs], device=device, dtype=t.float)
                              for h, m in zip(radical_hidden, entity_mask)]

        # generate relation candidate and classify & predict
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = copy.deepcopy(entity_type)
        element_structures = [
            [("ENTITY", index) for index in range(len(entity_type[i]))]
            for i in range(b)
        ]
        element_is_model_path = [
            [True] * len(entity_type[i])
            for i in range(b)
        ]
        num, layer = 0, 1
        result = [[] for _ in range(b)]

        while layer <= max_layer:
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]  # 产生候选对
            if not has_candidate(candidates):
                break
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in
                         range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in
                         range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(
                t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model

                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat(
                    [all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                     for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])

                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, op1_rad, sep_emb, op2_emb, op2_hid,
                     op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, op1_rad, sep_emb, op2_emb, op2_hid,
                     op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, op1_rad, sep_emb, op2_hid, op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                candidate_type_emb = self.type_trans(candidate_type_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, op1_rad, sep_emb, op2_hid, op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc
            assert candidate_mask.shape == predict.shape
            predict = predict * candidate_mask
            pmodel = self.cls(trans_out)  # b x mc
            assert candidate_mask.shape == pmodel.shape
            pmodel = pmodel * candidate_mask
            pllm = t.zeros_like(pmodel)
            llm_hit = t.zeros_like(pmodel, dtype=t.bool)
            llm_structured_hit = t.zeros_like(pmodel, dtype=t.bool)
            llm_legacy_hit = t.zeros_like(pmodel, dtype=t.bool)
            if pllm_scores is None:
                pllm_scores = [None] * b

            for i in range(b):
                sid_map = pllm_scores[i] if isinstance(pllm_scores, (list, tuple)) and i < len(
                    pllm_scores) else None
                if not isinstance(sid_map, dict):
                    continue
                has_structured_scores = (
                    sid_map.get(("__META__", "llm_score_format"))
                    == "recursive_entity_index_v1"
                )

                for j in range(len(candidates[i])):
                    rel = candidates[i][j][0]
                    op1 = int(candidates[i][j][1][0])
                    op2 = int(candidates[i][j][1][1])
                    if op1 >= len(element_structures[i]) or op2 >= len(element_structures[i]):
                        raise ValueError(
                            "candidate operand index exceeds runtime structure list: {} {} {}".format(
                                rel, op1, op2
                            )
                        )
                    structure_key = (
                        "RELATION",
                        rel,
                        element_structures[i][op1],
                        element_structures[i][op2],
                    )
                    legacy_key = (rel, op1, op2)

                    if structure_key in sid_map:
                        sc = sid_map[structure_key]
                        llm_structured_hit[i, j] = True
                    elif not has_structured_scores and legacy_key in sid_map:
                        sc = sid_map[legacy_key]
                        llm_legacy_hit[i, j] = True
                    else:
                        continue
                    if sc is not None:
                        try:
                            sc = float(sc)
                        except Exception:
                            sc = 0.0
                        if sc < 0.0:
                            sc = 0.0
                        elif sc > 1.0:
                            sc = 1.0

                        pllm[i, j] = sc
                        llm_hit[i, j] = True
            pfinal = pmodel.clone()
            llm_admitted = llm_hit & (pllm >= llm_threshold)
            if fusion_mode == "weighted":
                pfinal[llm_hit] = alpha * pmodel[llm_hit] + (1 - alpha) * pllm[llm_hit]
            elif fusion_mode == "add":
                pfinal[llm_admitted] = t.maximum(
                    pmodel[llm_admitted], pllm[llm_admitted]
                )
            elif fusion_mode == "compare":
                # Preserve the supervised model's layer-wise search.  LLM
                # scores only raise the score of an exact candidate structure; the
                # usual threshold and per-operand winner selection below then
                # decide which relation is propagated to the next layer.
                pfinal[llm_hit] = t.maximum(
                    pmodel[llm_hit], pllm[llm_hit]
                )
            self.last_llm_fusion_stats.append({
                "layer": layer,
                "fusion_mode": fusion_mode,
                "candidate_count": int(candidate_mask.sum().item()),
                "llm_hit_count": int(llm_hit.sum().item()),
                "llm_structured_hit_count": int(llm_structured_hit.sum().item()),
                "llm_legacy_hit_count": int(llm_legacy_hit.sum().item()),
                "llm_ge_05_count": int((llm_hit & (pllm >= 0.5)).sum().item()),
                "llm_admitted_count": int(llm_admitted.sum().item()),
            })

            predict = pfinal
            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])

                model_eligible = [
                    element_is_model_path[i][int(candidate[1][0])]
                    and element_is_model_path[i][int(candidate[1][1])]
                    for candidate in candidates[i]
                ]
                model_selected_indices = _top_model_candidates_by_operand(
                    candidates[i],
                    pmodel[i],
                    threshold,
                    eligible=model_eligible,
                )
                selected_indices = _select_scratch_candidates(
                    candidates[i],
                    pmodel[i],
                    predict[i],
                    threshold,
                    fusion_mode,
                    llm_admitted=llm_admitted[i],
                    model_eligible=model_eligible,
                )
                for j in sorted(selected_indices):
                    idx = all_elem_hidden[i].shape[0] - 1
                    all_elem_hidden[i][idx] = trans_out[i][j]
                    all_radical_hidden[i][idx] = trans_out[i][j]
                    all_elem_hidden[i] = do_pad(all_elem_hidden[i], max_l=idx + 2, pad_elem=0)
                    all_radical_hidden[i] = do_pad(all_radical_hidden[i], max_l=idx + 2, pad_elem=0)
                    type2index[i][candidates[i][j][0]][-1].append(idx)
                    index2type[i].append(candidates[i][j][0])
                    if idx != len(element_structures[i]):
                        raise ValueError(
                            "runtime element index/structure index mismatch: {} vs {}".format(
                                idx, len(element_structures[i])
                            )
                        )
                    op1, op2 = candidates[i][j][1]
                    element_structures[i].append((
                        "RELATION",
                        candidates[i][j][0],
                        element_structures[i][int(op1)],
                        element_structures[i][int(op2)],
                    ))
                    element_is_model_path[i].append(
                        fusion_mode != "add" or j in model_selected_indices
                    )
                    result[i].append(candidates[i][j])

            layer += 1

        return result

    def predict_guided(self, x, x_mask, entity_index, entity_type, elements, relations,
                       threshold=0.5, max_layer=6, pllm_scores=None,
                       fusion_mode="model"):
        if fusion_mode not in ("model", "compare"):
            raise ValueError("unsupported guided fusion mode: {}".format(fusion_mode))
        self.last_llm_fusion_stats = []
        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # encode input words
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        radical_encoder_output = self.radical_encoder(x)
        hs = encoder_output.shape[-1]

        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        radical_ehidden = [
            radical_encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h
        all_radical_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                              if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                              for h, m in zip(radical_ehidden, entity_mask)]
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)  ##########
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        gold_element_structures = []
        for i in range(b):
            structures = [
                ("ENTITY", index) for index in range(len(entity_type[i]))
            ]
            for relation, operands in elements[i][len(entity_type[i]):]:
                op1, op2 = int(operands[0]), int(operands[1])
                if op1 >= len(structures) or op2 >= len(structures):
                    raise ValueError(
                        "gold operand index exceeds guided structure list: {} {} {}".format(
                            relation, op1, op2
                        )
                    )
                structures.append((
                    "RELATION",
                    relation,
                    structures[op1],
                    structures[op2],
                ))
            gold_element_structures.append(structures)
        if pllm_scores is None:
            pllm_scores = [None] * b
        num, layer = 0, 1
        result = [[] for _ in range(b)]
        gold = [[] for _ in range(b)]
        all_candidate = [[] for _ in range(b)]
        all_prob = [[] for _ in range(b)]

        while layer <= max_layer:
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
            if not has_candidate(candidates):
                break
            for i_ in range(b):
                all_candidate[i_].append(candidates[i_])
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]  # 这里？
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, op1_rad, sep_emb, op2_emb, op2_hid,
                     op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, op1_rad, sep_emb, op2_emb, op2_hid,
                     op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type   m没有embedding
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, op1_rad, sep_emb, op2_hid, op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op1_rad = t.cat([all_radical_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_rad = t.cat([all_radical_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)
                op1_rad, op2_rad = self.hid_trans(op1_rad), self.hid_trans(op2_rad)
                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, op1_rad, sep_emb, op2_hid, op2_rad],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc
            assert candidate_mask.shape == predict.shape
            predict = predict * candidate_mask
            pmodel = predict
            pfinal = pmodel.clone()
            llm_hit = t.zeros_like(pmodel, dtype=t.bool)
            llm_structured_hit = t.zeros_like(pmodel, dtype=t.bool)

            if fusion_mode == "compare":
                for i in range(b):
                    sid_map = (
                        pllm_scores[i]
                        if isinstance(pllm_scores, (list, tuple)) and i < len(pllm_scores)
                        else None
                    )
                    if not isinstance(sid_map, dict):
                        continue
                    for j in range(len(candidates[i])):
                        relation = candidates[i][j][0]
                        op1 = int(candidates[i][j][1][0])
                        op2 = int(candidates[i][j][1][1])
                        if (
                            op1 >= len(gold_element_structures[i])
                            or op2 >= len(gold_element_structures[i])
                        ):
                            raise ValueError(
                                "guided candidate operand exceeds gold structure list: {} {} {}".format(
                                    relation, op1, op2
                                )
                            )
                        structure_key = (
                            "RELATION",
                            relation,
                            gold_element_structures[i][op1],
                            gold_element_structures[i][op2],
                        )
                        if structure_key not in sid_map:
                            continue
                        try:
                            score = float(sid_map[structure_key])
                        except Exception:
                            score = 0.0
                        score = min(1.0, max(0.0, score))
                        pfinal[i, j] = max(float(pmodel[i, j].item()), score)
                        llm_hit[i, j] = True
                        llm_structured_hit[i, j] = True

            self.last_llm_fusion_stats.append({
                "layer": layer,
                "fusion_mode": fusion_mode,
                "candidate_count": int(candidate_mask.sum().item()),
                "llm_hit_count": int(llm_hit.sum().item()),
                "llm_structured_hit_count": int(llm_structured_hit.sum().item()),
            })

            for i in range(b):
                result[i].append([])
                gold[i].append([])
                all_prob[i].append(pfinal[i].tolist())

                selected_indices = _top_model_candidates_by_operand(
                    candidates[i], pfinal[i], threshold
                )

                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])

                for j in range(len(candidates[i])):
                    # Guided inference must use exactly the same current-layer
                    # winner policy as standard inference.  Gold relations are
                    # injected only into the state used to build the next layer.
                    if j in selected_indices:
                        result[i][-1].append(candidates[i][j])

                    if candidates[i][j] in elements[i]:
                        gold[i][-1].append(candidates[i][j])
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        all_radical_hidden[i][idx] = trans_out[i][j]
                        type2index[i][candidates[i][j][0]][-1].append(idx)
            layer += 1
        return result, gold, all_candidate, all_prob

    def forward_weight(self, x, x_mask, entity_index, entity_type, elements, relations, relation_weight):
        """
        :param x: input words ids b x s
        :param x_mask: input mask b x s
        :param entity_index: b x ?s x 2, start and end index of all entity in sentences of a batch
        :param entity_type: b x ?s, entity types in sentences of a batch
        :param elements: b x ?n, element of all sentences in a batch, [(type: str, (op_idx, ...))]
        :param relations: r, contain type and ops types (type: str, (op_type))
        :param relation_weight: weight of relations, dict(relation, weight)
        :return:
        """
        b, device = x.shape[0], x.device
        # num of elem in each sentence
        elem_lens = [len(elem) for elem in elements]  # b x ?s
        # num of words in each entity of each sentence
        entity_words_len = [[entity_index[i][j][1] - entity_index[i][j][0] + 1
                             for j in range(len(entity_index[i]))] for i in range(b)]
        max_elem_len, max_ew_len = max(elem_lens), max([max(s) if s else 0 for s in entity_words_len])

        # encode input words
        encoder_output = self.encoder(x, attention_mask=x_mask)[0]  # b x l x h
        hs = encoder_output.shape[-1]
        # pad entity word hidden state to max word num for each sentence
        # entity_hidden = [t.cat([do_pad(
        #     encoder_output[i][t.range(entity_index[i][j][0], entity_index[i][j][1], dtype=t.long, device=device)],
        #     max_l=max_ew_len, pad_elem=0).unsqueeze(0) for j in range(len(entity_index[i]))
        #                         ]) for i in range(b)]  # b x ?s x mw x h*2
        entity_hidden = [
            encoder_output[i][t.tensor(
                [list(range(entity_index[i][j][0], 1 + entity_index[i][j][1])) + [0] * (
                        max_ew_len - entity_words_len[i][j])
                 for j in range(len(entity_index[i]))], dtype=t.long, device=device)] for i in
            range(b)]  # b x ?s x mw x h
        entity_mask = [t.tensor([[0] * l + [1] * (max_ew_len - l) for l in entity_words_len[i]],
                                dtype=t.bool, device=device) for i in range(b)]  # b x ?s x mw
        # get entity hidden for each sentence by Attention, then padding to max elem len
        all_elem_hidden = [do_pad(self.attention(h, m), max_l=max_elem_len + 1, pad_elem=0)
                           if h.shape[0] else t.zeros([max_elem_len + 1, hs], device=device, dtype=t.float)
                           for h, m in zip(entity_hidden, entity_mask)]  # b x me x h

        # generate candidates and classify
        pre_candidates = [[] for _ in range(b)]
        type2index = generate_type2index(list(self.type2id.keys()), entity_type)
        index2type = [[elements[i][j][0] for j in range(elem_lens[i])] for i in range(b)]
        loss, num, layer = None, 0, 1
        # candidates = [generate_candidate1(relations, type2index[i], pre_candidates[i])
        #               for i in range(b)]
        candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]

        while has_candidate(candidates):
            c_lens = [len(c) for c in candidates]
            max_c_len = max(c_lens)  # mc
            candidate_label = [[c in elements[i] for c in candidates[i]] + [False] * (max_c_len - c_lens[i])
                               for i in range(b)]
            candidate_label = t.tensor(candidate_label, dtype=t.float, device=device)
            # if candidate_label.sum() == 0:
            #     break
            candidate_weight = [[relation_weight[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                for i in range(b)]
            candidate_weight = t.tensor(candidate_weight, dtype=t.float, device=device)
            candidate_mask = t.tensor([[1] * c_lens[i] + [0] * (max_c_len - c_lens[i]) for i in range(b)],
                                      dtype=t.float, device=device)
            # print('train:', layer, '|', candidate_label.sum().item(), '|', candidate_mask.sum().item())
            candidate_type_ids = [[self.type2id[c[0]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                                  for i in range(b)]  # b x mc
            op1_type_ids = [
                [self.type2id[index2type[i][c[1][0]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op2_type_ids = [
                [self.type2id[index2type[i][c[1][1]]] for c in candidates[i]] + [0] * (max_c_len - c_lens[i])
                for i in range(b)]
            op1_index = [[c[1][0] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            op2_index = [[c[1][1] for c in candidates[i]] + [0] * (max_c_len - c_lens[i]) for i in range(b)]
            cls_index = [[self.cls_id] * max_c_len for _ in range(b)]
            sep_index = [[self.sep_id] * max_c_len for _ in range(b)]

            candidate_type_emb = self.type_embedding(t.tensor(candidate_type_ids, dtype=t.long, device=device))
            if self.trans_type == 0:  # whole model
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            elif self.trans_type == 1:  # w/o pos
                op1_emb = self.type_embedding(t.tensor(op1_type_ids, dtype=t.long, device=device))
                op2_emb = self.type_embedding(t.tensor(op2_type_ids, dtype=t.long, device=device))
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                # print(candidate_type_emb.shape, op1_emb.shape, op1_hid.shape)
                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_emb, op1_hid, sep_emb, op2_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            elif self.trans_type == 2:  # w/o op type
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
                candidate_hid = self.pos_encoder(candidate_hid)
            else:
                op1_hid = t.cat([all_elem_hidden[i][t.tensor(op1_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                op2_hid = t.cat([all_elem_hidden[i][t.tensor(op2_index[i], dtype=t.long, device=device)].unsqueeze(0)
                                 for i in range(b)])
                cls_emb = self.type_embedding(t.tensor(cls_index, dtype=t.long, device=device))
                sep_emb = self.type_embedding(t.tensor(sep_index, dtype=t.long, device=device))

                candidate_type_emb = self.type_trans(candidate_type_emb)
                # op1_emb, op2_emb = self.type_trans(op1_emb), self.type_trans(op2_emb)
                cls_emb, sep_emb = self.type_trans(cls_emb), self.type_trans(sep_emb)
                op1_hid, op2_hid = self.hid_trans(op1_hid), self.hid_trans(op2_hid)

                candidate_hid = t.cat(
                    [cls_emb, candidate_type_emb, sep_emb, op1_hid, sep_emb, op2_hid],
                    dim=-1).reshape([b * max_c_len, -1, hs])
            trans_out = self.DAG_trans(candidate_hid.float())[:, 0]  # b*mc x h
            trans_out = trans_out.reshape([b, max_c_len, -1])
            predict = self.cls(trans_out)  # b x mc

            t_loss = F.binary_cross_entropy(predict, candidate_label, reduce=False, )
            t_loss = t.sum(t_loss * candidate_weight)
            loss = t_loss if not loss else (t_loss + loss)
            num += candidate_mask.sum().item()

            for i in range(b):
                for tp in list(self.type2id.keys()):
                    type2index[i][tp].append([])
                for j in range(c_lens[i]):
                    if candidate_label[i][j]:
                        idx = elements[i].index(candidates[i][j])
                        all_elem_hidden[i][idx] = trans_out[i][j]
                        type2index[i][candidates[i][j][0]][-1].append(idx)
            candidates = [generate_candidate2(relations, type2index[i]) for i in range(b)]
        return loss / num if loss else t.tensor(0)




