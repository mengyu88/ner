# DiFiNet: Boundary-Aware Semantic Differentiation and Filtration Network for Nested Named Entity Recognition

## Overview

This is the code for [DiFiNet: Boundary-Aware Semantic Differentiation and Filtration Network for Nested Named Entity Recognition](https://openreview.net/forum?id=zAig3Mmy1v), accepted by ACL 2024.

![1703063577738](images/model.png)

## Requirements

```
GPU=NVIDIA A100 Tensor Core
beautifulsoup4==4.9.3
FastNLP==1.0.1
fitlog==0.9.15
nltk==3.8.1
numpy==1.24.4
pandas==1.1.3
sparse==0.14.0
torch==1.13.1+cu117
torch_scatter==2.0.9
tqdm==4.65.0
transformers==4.20.1
```

## Quick Deployment (Linux, Python 3.11)

```bash
git clone https://github.com/AONE-NLP/DiFiNet.git
cd DiFiNet
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip "setuptools<81" wheel
pip install -r requirements
```

### If HuggingFace is not reachable

You can switch to ModelScope download fallback:

```bash
pip install modelscope
export USE_MODELSCOPE=1
```

Then run training as usual. Model names are mapped automatically:
- `roberta-base` -> `AI-ModelScope/roberta-base`
- `bert-large-cased` -> `AI-ModelScope/bert-large-cased`

Note: `genia` uses `dmis-lab/biobert-v1.1`. If this model is unavailable on ModelScope, use either:
- local model path via `--model_name /path/to/model`
- HF mirror via `HF_ENDPOINT`

If you have a reachable HF mirror endpoint (for example wisemode), you can also use:

```bash
export HF_ENDPOINT=<your_mirror_endpoint>
```

You can also pass an explicit ModelScope id:

```bash
python train.py -d ace2005 --model_name ms://AI-ModelScope/roberta-base
```

## Preprocess your datasets
Put ACE datasets in the `preprocess/data` directory, following a similar structure as demonstrated in [CNN_Nested_NER](https://github.com/yhcc/CNN_Nested_NER) 

For GENIA dataset, we use the version from [W2NER](https://github.com/ljynlp/W2NER)
Dataset split files are provided in `preprocess/splits/`.

Examples:

```bash
# ACE2004
python preprocess/proAce04.py -i preprocess/data/ace_multilang_tr/data -o preprocess/outputs/ace2004

# ACE2005
python preprocess/proAce05.py -i preprocess/data/ace05/data -o preprocess/outputs/ace2005

# GENIA
python preprocess/proGenia.py -i preprocess/data/GENIAcorpus3.02.merged.fixed.xml -o preprocess/outputs/genia

# Weibo NER (CoNLL BIO -> jsonlines)
python preprocess/proWeibo.py -i data/weibo -o preprocess/outputs/weibo
```

On A10/A100-like GPUs, Weibo training may need smaller per-step batch size to avoid OOM.
Recommended:

```bash
bash train_arg_weibo.sh
```

## Train

   ```
   bash train_arg_{dataset}.sh
   ```
The item {dataset} can be replaced with "04", "05", "genia" or "weibo". The experiment results can be found in the directory `logs` after initiating the training process.


## Citation
```
@inproceedings{cai-etal-2024-difinet,
    title = "{D}i{F}i{N}et: Boundary-Aware Semantic Differentiation and Filtration Network for Nested Named Entity Recognition",
    author = "Cai, Yuxiang  and
      Liu, Qiao  and
      Gan, Yanglei  and
      Lin, Run  and
      Li, Changlin  and
      Liu, Xueyi  and
      Luo, Da  and
      Jiaye, Yang",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.349",
    pages = "6455--6471"
}
```
## Have any Questions？Please email cyx_yyy at foxmail dot com


## Acknowledge
The code of [CNN_Nested_NER](https://github.com/yhcc/CNN_Nested_NER)
