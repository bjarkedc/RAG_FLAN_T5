# Retrieval Augmented Generation with FlanT5, BM25, and TF-IDF

## Overview

This project explores the combination of various retrieval and generation techniques to enhance natural language understanding and generation. The primary components include Retrieval Augmented Generation (RAG) using FlanT5, and traditional information retrieval methods such as BM25.

## Features

- **FlanT5:** Leveraging FlanT5 for retrieval augmented generation, a state-of-the-art model that combines T5 with retrieval mechanisms.
- **BM25:** Using BM25, a classic probabilistic information retrieval model, for efficient document retrieval based on keyword matching.

## Getting Started

### Prerequisites

- Python 3.6 or later
- Dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Running the finetuning
Performing a hpyerparameter search based on the code provided in https://huggingface.co/robvanderg/flan-t5-base-starwars to do mlm.

A Weights & Biases ([wandb](https://wandb.ai/home)) account is needed.

Unzip the wookipedia.7z file in the root of the directory. 

```bash
cd fine_tune_t5/
python run_t5_mlm_torch.py --train_file ../starwarsfandomcom-20200223.txt.cleaned.tok.uniq.txt --output_dir flan-t5-base-starwars --validation_split_percentage 1 --model_name_or_path google/flan-t5-base --max_seq_length 512 --do_train --do_eval
```

### Testing models
To test the models trained with the hyperparameter search, run:

```bash
cd fine_tune_t5/
python test_models.py
```
