"""
Merges the trained LoRA adapter into the base model and exports a GGUF file
that LM Studio can load directly.

Run after train.py has finished and produced ./lora_adapter.
"""

from unsloth import FastLanguageModel

MODEL_PATH = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_DIR = "lora_adapter"
GGUF_OUTPUT_NAME = "llama3.2-3b-patient-guard"       # output filename prefix
QUANTIZATION = "q4_k_m"                              # good quality/size tradeoff; try q8_0 for higher fidelity

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,   # load full precision for a clean merge
)

# Load and merge the LoRA adapter
model.load_adapter(ADAPTER_DIR)

# Save merged model directly to GGUF (Unsloth handles the llama.cpp conversion)
model.save_pretrained_gguf(
    GGUF_OUTPUT_NAME,
    tokenizer,
    quantization_method=QUANTIZATION,
)

print(f"GGUF file written: {GGUF_OUTPUT_NAME}-{QUANTIZATION}.gguf")
print("Drag this file into your LM Studio models folder, refresh, and load it.")
