from typing import List
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer
from fastNLP import Vocabulary
from fastNLP.embeddings import CNNCharEmbedding

import sys
sys.path.append("./")
from alphabet import Alphabet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
radical_path = os.path.join(BASE_DIR, "lexicon/chaizi/chaizi-ft.txt")
char_info = dict()
with open(radical_path, "r") as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="Loading {}".format(radical_path)):
        char, info = line.split("\t", 1)
        char_info[char] = info.replace("\n", "").split("\t")


def char2radical(c):
    if c in char_info.keys():
        c_info = char_info[c]
        return list(c_info[0])
    return ["○"]


def construct_radical_vocab_from_vocab( char_vocab: Vocabulary, include_word_start_end=True):
    radical_vocab = Vocabulary()
    for char, _ in char_vocab:
        # if not char_vocab._is_word_no_create_entry(char):
        radical_vocab.add_word_lst(char2radical(char))
    if include_word_start_end:
        radical_vocab.add_word_lst(["<bow>", "<eow>"])
    return radical_vocab


def alphabet2vocab(phabet: Alphabet):
    vocab = Vocabulary()
    for word, _ in phabet.iteritems():
        vocab.add_word(word)
    return vocab





class RadicalViewFeature(nn.Module):
    def __init__(
        self,config,
        word_alphabet: Alphabet,
        filter_nums: List[int] = [50,30,20] ,
        kernel_sizes: List[int] = [5,3,1] ,
        # kernel_sizes: List[int] = [9,7,5],   #
        # filter_nums: List[int] = [40,30,20],
        # kernel_sizes: List[int] = [11,7,3]  ,
        pool_method: str = "max",
        include_word_start_end: bool = True,
    ):

        super(RadicalViewFeature, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.radical_feature_dim = config.radical_feature_dim
        self.pooler_dim = config.pooler_dim
        for kernel in kernel_sizes:
            assert kernel % 2 == 1, "Only odd kernel is allowed."
        assert pool_method in ("max", "avg")
        self.pool_method = pool_method
        self.include_word_start_end = include_word_start_end
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

        # Build char vocab

        self.chars_vocab = alphabet2vocab(word_alphabet)
        self.radical_vocab = construct_radical_vocab_from_vocab(self.chars_vocab)
        max_radical_nums = max(map(lambda x: len(char2radical(x[0])), self.chars_vocab))
        if include_word_start_end:
            max_radical_nums += 2
        self.char_pad_index = self.radical_vocab.padding_idx
        chars_to_radicals_embedding = torch.full(
            (len(self.chars_vocab), max_radical_nums),
            fill_value=self.char_pad_index,
        ).long()

        word_lengths = torch.zeros(len(self.chars_vocab)).long()
        for word, index in self.chars_vocab:
            word = char2radical(word)
            if self.include_word_start_end:
                word = ["<bow>"] + word + ["<eow>"]
            chars_to_radicals_embedding[index, : len(word)] = torch.LongTensor(
                [self.radical_vocab.to_index(c) for c in word]
            )
            word_lengths[index] = len(word)

        self.register_buffer(
            "chars_to_radicals_embedding",
            chars_to_radicals_embedding,
            persistent=False,
        )
        self.register_buffer("word_lengths", word_lengths, persistent=False)

        self.radical_embedding = nn.Embedding(
            len(self.radical_vocab), self.radical_feature_dim
        )
        self.radical_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.radical_feature_dim,
                    filter_nums[i],
                    kernel_size=kernel_sizes[i],
                    bias=True,
                    padding=kernel_sizes[i] // 2,
                )
                for i in range(len(kernel_sizes))
            ]
        )

        self.pooler = nn.Sequential(
            nn.Tanh(),
            nn.Linear(sum(filter_nums), self.pooler_dim),
        )


    def forward(self, words):
        decoded_sentences = []
        batch_size, max_len = words.size()
        for i in range(batch_size):
            sentence = []
            for j in range(max_len):
                idx = words[i, j].item()
                if idx != 0:
                    decoded_sentence = self.tokenizer.convert_ids_to_tokens([idx])[0]
                    index = self.chars_vocab.to_index(decoded_sentence)
                    if index is not None:
                        sentence.append(index)
                    else:
                        sentence.append(0)
            decoded_sentences.append(sentence)
        padded_sentences = [
            sentence + [0] * (max_len - len(sentence)) if len(sentence) < max_len else sentence[:max_len] for
            sentence in decoded_sentences]

        tensor_sentences = torch.tensor(
            padded_sentences,
            device=self.chars_to_radicals_embedding.device,
        )

        chars = self.chars_to_radicals_embedding[tensor_sentences]
        word_lengths = self.word_lengths[tensor_sentences]
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]

        chars_masks = chars.eq(self.char_pad_index)

        chars = self.radical_embedding(chars)
        reshaped_chars = chars.reshape(batch_size * max_len, max_word_len, -1)
        reshaped_chars = reshaped_chars.transpose(1, 2)  # B' x E x M
        conv_chars = [
            conv(reshaped_chars)
            .transpose(1, 2)
            .reshape(batch_size, max_len, max_word_len, -1)
            for conv in self.radical_convs
        ]
        # B x max_len x max_word_len x sum(filters)
        conv_chars = torch.cat(conv_chars, dim=-1).contiguous()
        # conv_chars = self.radical_cnn_feature(chars)

        if self.pool_method == "max":
            # batch_size x max_len x sum(filters)
            conv_chars = conv_chars.masked_fill(
                chars_masks.unsqueeze(-1), float("-inf")
            )
            chars, _ = torch.max(conv_chars, dim=-2)
        else:
            conv_chars = conv_chars.masked_fill(chars_masks.unsqueeze(-1), 0)
            chars = (
                torch.sum(conv_chars, dim=-2)
                / chars_masks.eq(False).sum(dim=-1, keepdim=True).float()
            )

        return self.pooler(chars)  # [b, l, dim]
