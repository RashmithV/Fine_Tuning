import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-3-1b-it"   # or local path
ADAPTER_PATH = "./qlora-out/checkpoint-216"  # NOTE: ./ is important

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)

# 🔑 THIS must be a local path
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    is_trainable=False
)

model.eval()

prompt = "### What is the best investment strategy for long-term growth?\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))