import torch
import warnings
import transformers
import pandas as pd
import bitsandbytes as bnb
import yaml
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

with open("lora_config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_name_or_path = config["model_name_or_path"]
auth_token = config["auth_token"]
rank = config["rank"]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config["bnb_config"]["load_in_4bit"],
    bnb_4bit_use_double_quant=config["bnb_config"]["bnb_4bit_use_double_quant"],
    bnb_4bit_quant_type=config["bnb_config"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=torch.bfloat16,
)

warnings.filterwarnings("ignore")

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit=True,
    device_map="cuda:0",
    trust_remote_code=True,
    token=auth_token,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    token=auth_token,
    # model_max_length=config.get('tokenizer_max_length', 512),
)

model = prepare_model_for_kbit_training(model)


def find_all_linear_names(model, bits):
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


target_modules = config["target_modules"]

lora_config = LoraConfig(
    r=rank,
    lora_alpha=rank * 2,
    target_modules=target_modules,
    lora_dropout=config["lora_dropout"],
    bias=config["lora_bias"],
    task_type=config["task_type"],
)

model = get_peft_model(model, lora_config)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)

data = pd.read_csv(config["data_csv_path"])
data = data.sample(frac=1).reset_index(drop=True)

data = Dataset.from_pandas(data)
data = data.map(lambda samples: tokenizer(samples[config["data_column_name"]]), batched=True)

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(**config["training_arguments"]),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

model.save_pretrained(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])
