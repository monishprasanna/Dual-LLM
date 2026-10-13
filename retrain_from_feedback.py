"""
retrain_from_feedback.py
Merges the original training dataset with new feedback examples captured from
detected leaks, then runs a short LoRA fine-tuning pass so those attack patterns
no longer fool the defender model.

Usage:
    python retrain_from_feedback.py

Or triggered automatically via the /api/retrain server endpoint.
"""

import json
import os
import sys

ORIGINAL_DATASET  = "patient_finetune_dataset.jsonl"
FEEDBACK_DATASET  = "feedback_dataset.jsonl"
MERGED_DATASET    = "merged_retrain_dataset.jsonl"
ADAPTER_DIR       = "lora_adapter"
MODEL_PATH        = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH    = 2048


# -----------------------------------------------------------------------
# Step 1 — Merge datasets
# -----------------------------------------------------------------------
def merge_datasets() -> int:
    """
    Combines patient_finetune_dataset.jsonl + feedback_dataset.jsonl into
    merged_retrain_dataset.jsonl.
    Returns the total number of examples.
    """
    lines = []

    for path in [ORIGINAL_DATASET, FEEDBACK_DATASET]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
        else:
            print(f"[Retrain] Note: {path} not found — skipping.")

    with open(MERGED_DATASET, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + "\n")

    print(f"[Retrain] Merged {len(lines)} total examples → {MERGED_DATASET}")
    return len(lines)


# -----------------------------------------------------------------------
# Step 2 — Fine-tune with Unsloth + LoRA
# -----------------------------------------------------------------------
def retrain() -> dict:
    total = merge_datasets()

    if total == 0:
        return {"status": "error", "message": "No training examples found."}

    feedback_count = 0
    if os.path.exists(FEEDBACK_DATASET):
        with open(FEEDBACK_DATASET, 'r', encoding='utf-8') as f:
            feedback_count = sum(1 for l in f if l.strip())

    if feedback_count == 0:
        return {
            "status": "nothing_to_train",
            "message": "No new feedback examples collected yet. "
                       "Trigger some attacks first so leaks get captured.",
        }

    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig
        import torch

        print("[Retrain] Loading base model (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )

        print("[Retrain] Attaching LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        dataset = load_dataset("json", data_files=MERGED_DATASET, split="train")

        def format_example(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        dataset = dataset.map(format_example)

        print(f"[Retrain] Training on {len(dataset)} examples "
              f"({feedback_count} new feedback + {total - feedback_count} original)...")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            args=SFTConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                # Fewer epochs than initial training — we only need to reinforce
                num_train_epochs=2,
                # Lower LR to avoid forgetting previously learned behaviour
                learning_rate=1e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=5,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="retrain_outputs",
                report_to="none",
            ),
        )

        trainer.train()

        model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)
        print(f"[Retrain] ✅ Updated LoRA adapter saved to ./{ADAPTER_DIR}")

        # Signal local_patient_model to reload its adapter on next request
        _write_reload_flag()

        return {
            "status":           "success",
            "total_examples":   total,
            "new_feedback":     feedback_count,
            "adapter_dir":      ADAPTER_DIR,
            "message":          f"Retraining complete on {feedback_count} new feedback "
                                f"examples + {total - feedback_count} original examples.",
        }

    except ImportError as e:
        # No GPU / Unsloth not installed — feedback saved for manual run later
        msg = (
            f"GPU/Unsloth not available ({e}). "
            f"Feedback saved to {FEEDBACK_DATASET}. "
            f"Run 'python retrain_from_feedback.py' on a GPU machine to apply."
        )
        print(f"[Retrain] {msg}")
        return {
            "status":         "no_gpu",
            "feedback_saved": feedback_count,
            "message":        msg,
        }

    except Exception as e:
        print(f"[Retrain] ERROR: {e}")
        return {"status": "error", "message": str(e)}


def _write_reload_flag():
    """Creates a flag file that tells local_patient_model.py to reload its adapter."""
    with open("reload_model.flag", "w") as f:
        f.write("1")


if __name__ == "__main__":
    result = retrain()
    print("\n--- Retrain Result ---")
    print(json.dumps(result, indent=2))
