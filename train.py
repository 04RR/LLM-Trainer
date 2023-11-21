import warnings
import transformers
import pandas as pd
from datasets import Dataset
from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

model_name_or_path = "lora/mathv1"
auth_token = ""

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit=True,
    device_map="cuda:0",
    trust_remote_code=True,
    token=auth_token,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    token=auth_token,
    # model_max_length=512,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


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

data = pd.read_csv("data/math/MathInstruct.csv")
data = data.sample(frac=0.1).reset_index(drop=True)

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
        save_steps=500,
        save_strategy="steps",
        logging_steps=100,
        output_dir="output/science_train/",
        optim="paged_adamw_8bit",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

model.save_pretrained("output/math/")
tokenizer.save_pretrained("output/math/")
