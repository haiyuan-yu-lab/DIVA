# :sparkles: DIVA: Disease-specific variant pathogenicity prediction using multimodal biomedical language models

## Overview

**DIVA** (**Di**sease-specific **va**riant pathogenicity prediction) is a deep learning framework that directly predicts specific types of diseases alongside the probability of deleteriousness for missense variants. It integrates information from two different modalities – protein sequence and disease-related textual annotations – encoded using two pre-trained language models and optimized within a contrastive learning paradigm designed to align variants with relevant diseases in the learned representation space. Our predictions can be accessed interactively at [https://diva.yulab.org/](https://diva.yulab.org/).

## System and hardware requirements

The model was trained and tested with the Linux operating system and a NVIDIA RTX A5000 GPU.

## Dependencies

```
biopython==1.79
numpy==1.24.4
pandas==1.4.4
PyYAML==6.0
requests>=2.32.2
scikit_learn==1.3.2
scipy==1.6.3
torch==1.13.1+cu117
transformers==4.28.1
```
See `requirements.txt`

## Installation

```
git clone https://github.com/haiyuan-yu-lab/DIVA.git
cd DIVA

conda create -n diva python=3.8
conda activate diva

pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Prepare input data
Example inputs are provided in `example_data/`

### Prepare variant inputs

* Context variants (required for inference): known disease variants that provide local disease contextual knowledge
* Variants of interest for prediction

### Prepare input feature for protein
* Sequence file (FASTA)
* Function annotation file

### Prepare disease vocabulary
* Vocabulary file
* Disease annotation file
* (Optional) Disease name mapping file

### AlphaMissense data (optional)

* Download AlphaMissense predictions (`AlphaMissense_aa_substitutions.tsv.gz`) from [here](https://console.cloud.google.com/storage/browser/dm_alphamissense)
* Process with `preprocess/prep_alphamissense.py`
* *Set `use_alphamissense=False` in CONFIG if choosing not to use AlphaMissense prediction scores*

## Running DIVA

The following scripts are intended to be run from the `dev/` directory.

### Model training

```
python -u disease_model_pipeline.py --config ./configs/dis_var_train_config.yaml --tensorboard True
```
For reproduction, we provide model weights in the "Releases" section. Please download  `best_model.pt` and place it in the `checkpoints/` directory. The released model was trained using [ESM2-150M](https://huggingface.co/facebook/esm2_t30_150M_UR50D) and [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) as pre-trained language model encoders.

### Inference

```
python -u disease_inference.py --config ./configs/dis_var_pred_config.yaml
```

#### Inference example

Please see `example_data/` directory for example data files and descriptions.

```
python -u disease_inference.py --config ./configs/example_pred.yaml
```
Expected running time: < 1min on single GPU (tested on NVIDIA RTX A5000)

The following files will be generated in the specified output directory:

* Disease specificity scores: `example_vars_pheno_score.tsv`
* Binary deleteriousness scores: `pred_example_vars_score.txt`
* Top-k (k=100 by default) disease-specificity predictions for input variants: `example_vars_topk.pkl`
* Embeddings in the shared variant-disease representation space (to compute disease-specificity score):
  * Variant embeddings `example_vars_pheno_pred_emb.npy`
  * Disease embeddings: `phenotype_emb.npy`

## Reference
> Liu, Yilin, David N. Cooper, and Haiyuan Yu. "Disease-specific variant pathogenicity prediction using multimodal biomedical language models." [bioRxiv (2025)](https://www.biorxiv.org/content/10.1101/2025.09.09.675184v1)