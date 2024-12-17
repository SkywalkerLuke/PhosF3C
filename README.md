# PhosF3C: A Feature Fusion Architecture with Fine-Tuned Protein Language Model and Conformer for Prediction of General Phosphorylation Site

PhosF3C is an advanced architecture designed to predict general phosphorylation sites in proteins. It leverages a fusion of a fine-tuned protein language model (ESM2) and a Conformer network to deliver accurate and efficient predictions. This repository provides the necessary tools to train, test, and use the model for phosphorylation site prediction.

## Table of Contents
- [Getting Started with this Repo](#getting-started-with-this-repo)
  - [Setup](#setup)
  - [Get ESM2 Model](#get-esm2)
  - [Get Model Weights](#get-model-weights)
  - [Set Your Data](#set-your-data)
  - [Set Your Prediction Config](#set-your-prediction-config)
- [Quick Start for Prediction](#quick-start-for-prediction)
  - [Use by Sequence](#use-by-sequence)
  - [Use by Dataset](#use-by-dataset)
- [Train Your Own Model](#train-your-own-model)
  - [Step 1: Train LoRA](#step-1-train-lora)
  - [Step 2: Train Conformer](#step-2-train-conformer)
- [Test Your Model](#test-your-model)

## Getting Started with this Repo

### Setup

To get started, install the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### get esm2

PhosF3C utilizes the ESM2 model for embedding protein sequences. You can load the model and weights with the following code:

```bash
import torch
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
```

This will download the pretrained ESM2 model and alphabet, which you can use for embedding protein sequences.

**NOTE:

### get model weights
Download the pretrained model weights for both LoRA and Conformer components.
These weights will be used for prediction or your own fine-tuning.
Download the  model here:
 

Remember to place them under ./model_ckp/lora and ./model_ckp/conformer respectively

### set your data
(only for using dataset)
Prepare your data for prediction. Ensure that your input sequences are in the required format (FASTA) without annotation. 
We set an expample prediction dataset, you can see it at ./data/data_predict_example.fasta
Remember to place your own data under ./data 

### set your prediction config
Before running predictions, configure your prediction settings. 
This includes defining the sequence types, input/output files, and other parameters relevant to your use case. Adjust these parameters according to the dataset and prediction task.
We also set an expample prediction config.
For those who wish to use whole architecture, use ./hparams/conformer.yaml
or 
For those who wish to only use the lora-fine-tuned esm use ./hparams/conformer.yaml
If you want to use your own config, place your yaml file under ./haprams 

## Quick start for prediction

### use by sequence

### use by dataset

```bash
python predict.py
```


## Train your own model

initiate your training config

### Step 1: Train lora

```bash
python train.py
```

### Step2 : Train conformer

## Test your model
```bash
python test.py
```
