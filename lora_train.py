import torch
import warnings
import transformers
import pandas as pd
import bitsandbytes as bnb
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

warnings.filterwarnings("ignore")

model_name_or_path = "stabilityai/stablelm-3b-4e1t"
auth_token = ""
rank = 16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

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
    # model_max_length=512,
)

# model.gradient_checkpointing_enable()
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


# target_modules = find_all_linear_names(model, 8)
# target_modules.append("lm_head")

target_modules = [
    # "gate_proj",
    "q_proj",
    "v_proj",
    # "o_proj",
    # "down_proj",
    # "up_proj",
    "k_proj",
    "lm_head",
]

config = LoraConfig(
    r=rank,
    lora_alpha=rank * 2,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)


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

data = pd.read_csv("no_robot/train_prompt.csv")
data = data.sample(frac=1).reset_index(drop=True)

data = Dataset.from_pandas(data)
data = data.map(lambda samples: tokenizer(samples["data"]), batched=True)

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        warmup_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_safetensors=True,
        save_steps=100,
        save_strategy="steps",
        logging_steps=50,
        output_dir="output/no_robot/",
        optim="paged_adamw_8bit",
        report_to=None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

model.save_pretrained("output/no_robot/")
tokenizer.save_pretrained("output/no_robot/")
