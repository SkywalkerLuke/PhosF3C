# PhosF3C
PhosF3C: A Feature Fusion Architecture with Fine-Tuned Protein Language Model and Conformer for prediction of general phosphorylation site

## Architecture

## Table of contents
-[Getting started with this ](#)
-[Quick start for prediction](#)
  *[use by sequence](*)
  *[use by dataset](*)
-[Train your own model](*)
  *[Step 1: Train lora](*)
  *[Step 2: Train conformer](*)
-[Test your model](*)


## Getting started with this repo

### set up
```bash
pip install -r requirement
```

### get esm2
```bash
import torch
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
```
### get model weights

### set your data

### set your prediction config

## Quick start for prediction

### use by sequence

### use by dataset

```bash
python predict.py
```
**NOTE:

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
