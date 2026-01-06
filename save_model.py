import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "google/gemma-3-1b-it"   # or local path
ADAPTER_PATH = "./qlora-adapter"
OUTPUT_PATH = "./gemma-3-1b-it-finetuned"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load base model in FP16 (NOT quantized)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)

# Load adapter and merge
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ Merged model saved to:", OUTPUT_PATH)
