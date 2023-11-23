# Train LLMs with qLoRA!

## Introduction
This repository contains scripts and configurations for training and merging models using the qLoRA method for efficient model training.

## Prerequisites
- Python 3.x
- PyTorch
- Transformers
- BitsAndBytes
- pandas
- YAML

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Configuration File (lora_config.yaml)
This YAML file contains configuration settings for the training process. Update the auth_token with your token and adjust other parameters as per your requirement.

## Training the Model (lora_train.py)
The lora_train.py script trains the model based on the configuration provided in lora_config.yaml. 
Please make sure the data you have is in the appropriate format and mention the column name that has the data in the config file.

To start the training process, make sure all the values in the lora_config.yaml file are correct and then run the training script:

```bash
python lora_train.py
```
The script will save the trained model in the specified output directory.

## Merging LoRA Layers (merge_lora.py)
The merge_lora.py script merges LoRA layers into a base model.

Before running the script, fill in the model_name_or_path, auth_token, out_folder_path, and lora_checkpoint_path in the script.

## Troubleshooting
If you encounter any issues, please check if your environment meets all prerequisites. For further assistance, create an issue in this repository.
