# GeoSyn-RE

This directory is the submission-oriented GNRED implementation of GeoSyn-RE.
It contains the supervised model, the syntax candidate generator, the LLM API
scorer, and their final score-fusion path.

## Quick smoke test

The following command runs a small deterministic example:

```bash
python examples/quick_test.py
```

The test uses one synthetic LTP-enriched record and the checked-in relation
schema. It does not download model weights, require a GPU, train a model, call
an LLM API, or use an API key. A successful run prints:

```text
quick test passed: generated one EQ candidate
```

This smoke test verifies only the deterministic syntax-evidence and candidate
generation stage. It is not a replacement for the full supervised-model and
LLM-assisted experiment.


## Environment

The recorded environment uses Python 3.6.13 and PyTorch 1.10.2. On a compatible
system:

```bash
conda env create -f environment.yml
conda activate geosyn-re
```

The full pipeline also requires the model resources described below and the
configuration of the corresponding LLM API. The deterministic quick test does
not require these resources.

### LTP and PyLTP

The syntax-analysis stage uses PyLTP to run Chinese POS tagging and dependency
parsing. The full-pipeline entry point expects the model files at
`assets/ltp/pos.model` and `assets/ltp/parser.model` (or paths supplied with
`--pos-model` and `--parser-model`). Download the LTP model files from the
official page:

<https://ltp.ai/download.html>

Extract the POS and parser model files and place them at the paths above. The
PyLTP project and its version information are at
<https://github.com/HIT-SCIR/pyltp>.

### BERT encoder

GeoSyn-RE uses Geo-BERT initialized from the public Chinese-BERT-wwm model
(`hfl/chinese-bert-wwm`) released by YMCui/HFL:

<https://github.com/ymcui/Chinese-BERT-wwm>

To download the public initialization model and save it in the directory
expected by this code, use the following command with the repository's Python
environment:

```bash
python -c "from transformers import BertTokenizer, BertModel; n='hfl/chinese-bert-wwm'; d='bert_geo'; BertTokenizer.from_pretrained(n).save_pretrained(d); BertModel.from_pretrained(n).save_pretrained(d)"
```

The command downloads the Chinese-BERT-wwm base model and saves it in
`bert_geo`.

### Data and file formats

The supervised runner reads the GNRED JSONL splits in `data/gnred/` and the
relation/type schema in `data/type3.json`. Each split contains one JSON record
per line, including a sentence, entity spans, and annotated relation elements.
The LTP candidate generator consumes LTP-enriched JSONL records containing
`sid`, `sentence`, `tokens`, `pos`, dependency arcs (`dep`), entities, and their
token alignment (`entity_map`). It writes candidate JSONL records with
`sid`, `sentence`, `entities`, and recursively structured `triples`.
The full pipeline writes intermediate stages and a run manifest under the
output directory.

The GNRED JSONL splits in `data/gnred/` are the authors' own dataset and are
provided with this repository for reproducing the reported experiments.

## Train the supervised model

```bash
python run_geosyn.py train \
  --device cuda:0 \
  --seed 66 \
  --batch-size 1 \
  --epochs 100 \
  --early-stop 20 \
  --output-dir checkpoints/gnred_seed66
```

The framework writes its best checkpoint as `model.pt` in the output directory.

## Run GeoSyn-RE

The reported experiments used GPT, DeepSeek, and Gemini. Keep API keys private;
do not commit them or place them in input files.

```bash
export OPENAI_API_KEY="your-key"

python run_geosyn.py pipeline \
  --checkpoint checkpoints/gnred_seed66/model.pt \
  --device cuda:0 \
  --threshold 0.60 \
  --api-provider openai \
  --api-model gpt-5.6 \
  --output-dir outputs/gnred_v29_openai
```

The command runs the complete LTP, LLM scoring, score conversion, and supervised
comparison pipeline. Final predictions and comparison audits are written under
`outputs/gnred_v29_openai/compare/`.

## License

The MIT license in `LICENSE` applies only to original GeoSyn-RE source code
whose authors have the right to license it. It does not grant rights to
third-party software, model weights, datasets, or other separately licensed
materials.
