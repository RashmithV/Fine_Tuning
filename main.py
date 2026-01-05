import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
model_name = "google/gemma-3-1b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit
    bnb_4bit_quant_type="nf4",      # NormalFloat4 (best quality)
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,                             # Rank
    lora_alpha=32,                   # Scaling
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="train.json")
def combine_prompt_response(example):
    example["text"] = f"### Question: {example['question']}\n### Answer: {example['answer']}"
    return example
dataset = dataset.map(combine_prompt_response)

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()  # Important!
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="./qlora-out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none"
)
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    args=training_args
)

trainer.train()

model.save_pretrained("qlora-adapter")
tokenizer.save_pretrained("qlora-adapter")
