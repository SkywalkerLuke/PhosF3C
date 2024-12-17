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

**NOTE**: the esm repo and checkpoint will be saved in `.cache/torch/hub/ `

### get model weights
Download the pretrained model weights for both LoRA and Conformer components.
These weights will be used for prediction or your own fine-tuning.
Download the  model here:
 

Remember to place them under `./model_ckp/lora` and `./model_ckp/conformer` respectively.

### set your data
(only for using dataset)
Prepare your data for prediction. Ensure that your input sequences are in the required format (FASTA) without annotation. 
We set an expample prediction dataset, you can see it at `./data/data_predict_example.fasta`.
For train and test, see them at `./data/data_train_test_example.fasta`.
Remember to place your own data under `./data`.

### set your prediction config (also for train and test)
Before running predictions, configure your prediction settings. 
This includes defining the sequence types, input/output files, and other parameters relevant to your use case. Adjust these parameters according to the dataset and prediction task.
We also set an expample prediction config.
For those who wish to use whole architecture, use `./hparams/conformer.yaml`and `./hparams/conformer.yaml` for just fine-tuned ESM2.
If you want to use your own config, place your yaml file under `./haprams`.

## Quick start for prediction



### Use by Dataset

If you have a dataset of protein sequences for batch prediction, you can configure the prediction parameters in a `yaml` configuration file. Here's how you can do it:

**Configure the prediction settings**: Edit the `yaml` configuration file (e.g., `conformer.yaml` or `lora.yaml`) to include the following parameters:

   - `predict_path`: The path to the file containing protein sequences for prediction.
   - `predict_result_path`: The path where the prediction results will be saved.
   - `threshold`: The threshold for predicting positive phosphorylation sites. We recommend setting this value to `0.5` (default value).
   - `type`: The type of phosphorylation site you wish to predict (should be list).
   - `device`: Set the device to use for prediction, either `cuda` for GPU or `cpu` for CPU.
   - `checkpoint`: model weights for lora or conformer (you can change for yours if you train your own model).

   Example of setiing in a `yaml` configuration file:
   ```yaml
  predict_path: ./data/
  predict_result_path: ./result/conformer
  threshold: 0.5
  type: ['S']
  device: cuda
   ```

**Run the prediction**: After configuring the yaml file, run the prediction script with the following command:

```bash
python predict.py --config_path ./hparams/conformer.yaml
```
Or for LoRA:
```bash
python predict.py --config_path ./hparams/lora.yaml
```
This will process the sequences from the predict_path, and the prediction results will be saved in the specified output folder (`./result/conformer` or `./result/lora`) as a JSON file. The JSON file will contain the predicted phosphorylation site information, including the predicted label (positive:1 or negative:0) and prediction score for each site.

**Example of output JSON format**:

```json
[
    {
        "id": "protein_1",
        "position": 123,
        "output": 0.85,
        "predict": 1
    },
    {
        "id": "protein_1",
        "position": 156,
        "output": 0.32,
        "predict": 0
    }
]
```

### use by sequence
To make predictions using individual sequences, refer to the `./example.py` script. You just need to place your protein sequence data into the `data` variable in the script, and the model will handle the rest.
The configuration settings (such as `predict_path`, `predict_result_path`, `threshold`, `type`, and `device`) are the same as those used for predicting on a dataset, as described below.
After that, run:
```bash
python example.py --config_path ./hparams/lora.yaml
```
## Train your own model

Before you start training your own model, you need to initiate your training configuration in  `yaml` file. This configuration file will specify all the necessary training information, such as batch size, learning rate, and more. Additionally, you can set up validation information (such as validation interval and patience) and specify where to save the logs and model weights.

The configuration file should include the following sections:

- **Training Information**: Set your batch size, learning rate, and other hyperparameters.
- **Validation Information**: Set parameters like validation interval and early stopping patience.
- **Saving Information**: Define the paths where logs and model weights will be saved.

For example, see the `lora.yaml` configuration file located at `./hparams/lora.yaml`. This file provides a template for training both the LoRA and Conformer models.

### Step 1: Train LoRA
After setting up the configuration, run the following command:


```bash
python train.py --config_path ./hparams/lora.yaml
```
This will start the LoRA training process, and the trained model will be saved as a `.pt` file in the `./model_ckp/lora` directory. Training logs will be saved in the `./log/lora` directory as a `.json` file. The log file will contain basic information such as the training step and validation performance at each step.
### Step2 : Train conformer
After training the LoRA model, the next step is to train the Conformer component. The configuration for Conformer is similar to that of LoRA, and you can refer to the `conformer.yaml` file located at `./hparams/conformer` for guidance.

In the Conformer training configuration, you will need to specify the checkpoint of the trained LoRA model by setting the `lora_checkpoint parameter`. This will ensure that the Conformer network is initialized with the fine-tuned LoRA model.

After that
Run the following command:

```bash
python train.py --config_path ./hparams/lora.yaml
```

## Test Your Model

Once you have trained your LoRA and Conformer models, you can evaluate their performance on a test dataset.
In the configuration file, you need to specify the test file and checkpoint of the model you want to test.


To run the test on your trained model, use the following command:

```bash
python test.py --config_path ./hparams/conformer.yaml
```
or
```bash
python test.py --config_path ./hparams/conformer.yaml
```

After running the test, the evaluation metrics will be saved in the log directory (e.g. `./log/conformer/test_log.json`). The test results will include various metrics such as Accuracy, AUC, Precision, Recall, and F1 Score.

Here is an example of the output JSON format for the test results:
```json
{
    "Accuracy": 0.7219,
    "AUC": 0.763,
    "MCC": 0.2574,
    "Precision": 0.2322,
    "Recall": 0.6472,
    "F1 Score": 0.3418,
    "Positive Accuracy": 0.6472,
    "Negative Accuracy": 0.7313
}
```
