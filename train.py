"""
Fine-tunes Llama-3.2-3B-Instruct on the patient_finetune_dataset.jsonl using
Unsloth + LoRA, running locally on your NVIDIA GPU.

Before running:
  1. Activate your environment: source unsloth-env/bin/activate
  2. Make sure patient_finetune_dataset.jsonl is in this same folder
     (run generate_dataset.py first if it isn't)
  3. Set MODEL_PATH below to point at your local Llama-3.2-3B-Instruct files
     (a folder containing config.json, tokenizer files, and safetensors weights)

Output: a LoRA adapter saved to ./lora_adapter, and (optionally) a merged
16-bit model saved to ./merged_model for GGUF conversion.
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

# -------------------------------------------------------------------
# CONFIG — edit these
# -------------------------------------------------------------------
MODEL_PATH = "unsloth/Llama-3.2-3B-Instruct"
DATASET_PATH = "patient_finetune_dataset.jsonl"
OUTPUT_ADAPTER_DIR = "lora_adapter"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True   # keeps VRAM usage low; fine for a 3B model even on 8GB cards

# -------------------------------------------------------------------
# 1. Load base model + tokenizer
# -------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,                 # auto-detect (bf16/fp16)
    load_in_4bit=LOAD_IN_4BIT,
)

# -------------------------------------------------------------------
# 2. Attach LoRA adapters
# -------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# -------------------------------------------------------------------
# 3. Load dataset and format with the Llama-3.2 chat template
# -------------------------------------------------------------------
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_example)

# -------------------------------------------------------------------
# 4. Train
# -------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=3,          # 3-5 as recommended; start with 3
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="training_outputs",
        report_to="none",
    ),
)

trainer.train()

# -------------------------------------------------------------------
# 5. Save the LoRA adapter
# -------------------------------------------------------------------
model.save_pretrained(OUTPUT_ADAPTER_DIR)
tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)
print(f"LoRA adapter saved to ./{OUTPUT_ADAPTER_DIR}")
print("Next: run merge_and_export.py to produce a GGUF file for LM Studio.")
