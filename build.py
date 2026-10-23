# -*- coding: gbk -*-
import sys
import numpy as np
from tqdm import tqdm
import re
from alphabet import Alphabet
import json
from tqdm import tqdm
def str2list(str, spec=None):
    str = re.sub("\s+", "", str)
    res = []
    i = 0
    while i < len(str):
        match = False
        if spec:
            for sp in spec:
                splen = len(sp)
                if i + splen <= len(str) and str[i : i + splen] == sp:
                    res.append(sp)
                    i += splen
                    match = True
                    break
        if not match:
            res.append(str[i])
            i += 1
    return res

def normalize_word(word):

    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += "0"
        else:
            new_word += char
    return new_word




def build_alphabet(input_file, word_alphabet, number_normalized=True):

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(input_file, "r", encoding="gbk") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading the file: {e}")
            raise

    words_longer_than_one = set()
    train_freq = {}
    NULLKEY = "<NULL>"
    biword_alphabet = set()
    char_alphabet = set()

    for line in tqdm(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        sentence = item.get("sentence", "")
        elements = item.get("element", [])
        entity = item.get("entity", [])

        sent = str2list(sentence, words_longer_than_one)

        for elem in elements:
            rel = elem[0]
            if train_freq.get(rel, -1) != -1:
                train_freq[rel] += 1
            else:
                train_freq[rel] = 1

        for widx, word in enumerate(sent):
            if number_normalized:
                word = normalize_word(word)
            word_alphabet.add(word)  # 使用传入的 word_alphabet 对象

            if widx < len(sent) - 1 and len(sent) > 2:
                nxtword = sent[widx + 1]
                if number_normalized:
                    nxtword = normalize_word(nxtword)
            else:
                nxtword = NULLKEY

            biword = word + nxtword
            biword_alphabet.add(biword)

            for char in word:
                char_alphabet.add(char)

    return word_alphabet, train_freq, biword_alphabet, char_alphabet