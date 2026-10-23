# Third-Party Notices

This file distinguishes original GeoSyn-RE source code from third-party
software, models, and data. The MIT license in `LICENSE` applies only to
original GeoSyn-RE source code that the authors are entitled to license.

## LTP and PyLTP

GeoSyn-RE uses the PyLTP interface for Chinese part-of-speech tagging and
dependency parsing. The repository contains a platform-specific PyLTP wheel
and LTP model files under `third_party/` and `assets/ltp/`. PyLTP is maintained
by HIT-SCIR:

<https://github.com/HIT-SCIR/pyltp>

The legacy LTP model download page is:

<https://ltp.ai/download.html>

These components remain subject to the upstream terms. In particular, the LTP
project describes different conditions for academic/non-commercial and
commercial use. Confirm the current terms on the linked upstream pages before
publishing a public mirror containing the model files or wheel. If those terms
do not permit redistribution, publish the source code with download
instructions instead of bundling the artifacts.

## Chinese character decomposition lexicon

The files under `lexicon/chaizi/` are third-party Chinese character
decomposition data. The included upstream notice identifies the data as
Creative Commons Attribution 3.0 (CC BY 3.0):

<https://creativecommons.org/licenses/by/3.0/>

The original notice and attribution requirements are retained in
`lexicon/chaizi/README.md` and `lexicon/chaizi/LICENSE`.

## Python dependencies

The runtime environment installs third-party packages listed in
`requirements.txt`, including NumPy, PyTorch, Transformers, fastNLP, tqdm,
pandas, openpyxl, jieba, and requests. Each package remains under its own
upstream license; consult the package metadata and the corresponding upstream
repository before redistribution. This repository does not relicense those
packages under MIT.

## BERT weights and GNRED data

The `bert_geo/` directory contains the Geo-BERT initialization resources
obtained from the public `hfl/chinese-bert-wwm` model:

<https://github.com/ymcui/Chinese-BERT-wwm>

<https://huggingface.co/hfl/chinese-bert-wwm>

The upstream Chinese-BERT-wwm repository identifies its software/model
distribution terms as Apache License 2.0. Retain the upstream attribution and
license notice when redistributing those model files. The repository's MIT
license does not relicense the BERT weights or tokenizer.

The GNRED JSONL files under `data/gnred/` are the authors' own dataset and are
included for the reported experiments. They are not covered by the MIT source
license; retain the dataset attribution and any reuse conditions specified by
the authors.

## External LLM services

The repository does not distribute an LLM or an LLM provider's SDK. The
optional API stage sends prompt text, source sentences, entity annotations,
and candidate relations to the provider selected by the user. OpenAI and
DeepSeek are external services with their own terms, privacy rules, model
availability, and usage charges. The current CLI implements OpenAI and
DeepSeek only; a Gemini result requires a separately documented and tested
client. The user is responsible for provider authorization and for checking
whether the input data may be transmitted.
