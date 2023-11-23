import torch
import warnings
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

model_name_or_path = "WizardLM/WizardCoder-3B-V1.0"
auth_token = "" # FILL IN DETAILS
out_folder_path = "" # FILL IN DETAILS
lora_checkpoint_path = "" # FILL IN DETAILS
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    token=auth_token,
    # model_max_length=512,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, return_dict=True, torch_dtype=torch.float16, device_map=device
)

peft_model = PeftModel.from_pretrained(model, lora_checkpoint_path)
peft_model = peft_model.merge_and_unload()

print("Models loaded.")

peft_model.save_pretrained(out_folder_path)
tokenizer.save_pretrained(out_folder_path)

print("Models saved.")
