import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer OOM from parallel processes

from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer

print ("update v11")
# NOTE: This script handles datasets from parse_llm_finetune_dataset.py which can use:
# - PROMPT_FORMAT="multi": Mixed Gemma/Alpaca/ChatML formats (all saved with "text" field)
# - PROMPT_FORMAT="alpaca": Alpaca format only (instruction/input/output fields)
# The script auto-detects format and applies formatting only if needed.

# Model configuration
max_seq_length = 4096  # Increased for article + claim
dtype = None
load_in_4bit = True

# Dataset configuration
dataset_percentage = 1.0 # Use 85% of dataset (set to 0.8 for 80%, 0.7 for 70%, etc.)
eval_dataset_percentage = 0.05 # Use 20% of eval dataset

# Load model
print("Loading Gemma-2 9B model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
print("Adding LoRA adapters...")
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
    use_rslora=False,
    loftq_config=None,
)

# Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    """
    Handle mixed format datasets from parse script:
    - If sample has "text" field populated (Gemma/ChatML): use it as-is
    - If sample has instruction/input/output (Alpaca): format it
    """
    texts = []
    
    # Get fields (may contain None values)
    existing_texts = examples.get("text", [None] * len(examples.get("instruction", [])))
    instructions = examples.get("instruction", [None] * len(existing_texts))
    inputs = examples.get("input", [None] * len(existing_texts))
    outputs = examples.get("output", [None] * len(existing_texts))
    
    # Process each sample
    for text, instruction, input_val, output in zip(existing_texts, instructions, inputs, outputs):
        if text is not None and text != "":
            # Already formatted (Gemma or ChatML format)
            texts.append(text)
        elif instruction is not None:
            # Alpaca format - needs formatting
            formatted_text = alpaca_prompt.format(instruction, input_val, output) + EOS_TOKEN
            texts.append(formatted_text)
        else:
            raise ValueError("Sample has neither 'text' nor 'instruction' field populated!")
    
    return {"text": texts}

# Print system info
import socket
output_dir = "/home/avramit/classifier/01122025/llmfinetune/hebrew_evidence_llm_lora_85p/"
print("\n" + "="*50)
print("TRAINING CONFIGURATION")
print("="*50)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Host IP: {socket.gethostbyname(socket.gethostname())}")
print(f"Model: unsloth/gemma-2-9b")
print(f"Output dir: {output_dir}")
print(f"Max seq length: {max_seq_length}")
print(f"Dataset %: {dataset_percentage*100}%")
print(f"4-bit quantization: {load_in_4bit}")
print(f"LoRA r: 16, alpha: 16")
print("="*50 + "\n")

# Load preprocessed dataset
print("Loading train dataset...")
train_dataset = load_from_disk("/home/avramit/classifier/01122025/datasets/llmfinetune/mafat/parse/train/")

# Apply dataset percentage filtering
if dataset_percentage < 1.0:
    original_size = len(train_dataset)
    subset_size = int(original_size * dataset_percentage)
    train_dataset = train_dataset.select(range(subset_size))
    print(f"Using {dataset_percentage*100}% of train dataset: {subset_size}/{original_size} examples")
else:
    print(f"Using full train dataset: {len(train_dataset)} examples")

# DEBUG: Check dataset structure
print("\n" + "="*50)
print("DEBUG: Dataset Structure Check")
print("="*50)
print(f"Dataset columns: {train_dataset.column_names}")
print(f"First sample keys: {train_dataset[0].keys()}")
print("\nFirst sample preview:")
for key in train_dataset[0].keys():
    value = train_dataset[0][key]
    if isinstance(value, str):
        print(f"  {key}: {value[:100] if len(value) > 100 else value}...")
    else:
        print(f"  {key}: {value}")
print("="*50 + "\n")

# Check if dataset already has "text" field
if "text" in train_dataset.column_names:
    print("✅ Dataset already has 'text' field - no formatting needed")
    first_text = train_dataset[0]["text"]
    if first_text and len(first_text) > 0:
        print(f"   Sample text preview: {first_text[:150]}...")
    else:
        print("   ⚠️ WARNING: 'text' field exists but is empty!")
elif "instruction" in train_dataset.column_names:
    print("📝 Dataset has Alpaca format - applying formatting...")
    print("   Using single process (num_proc=1) to avoid multiprocessing issues...")
    
    # Simplified formatting function for Alpaca (no mixed format check needed)
    def format_alpaca_simple(examples):
        texts = []
        for instruction, input_val, output in zip(examples["instruction"], examples["input"], examples["output"]):
            text = alpaca_prompt.format(instruction, input_val, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    
    train_dataset = train_dataset.map(format_alpaca_simple, batched=True, num_proc=1, remove_columns=["instruction", "input", "output"])
    print("   ✅ Formatting complete")
    print(f"   Final columns: {train_dataset.column_names}")
else:
    print("❌ ERROR: Dataset has neither 'text' nor 'instruction' fields!")
    print(f"   Available fields: {train_dataset.column_names}")
    raise ValueError("Incompatible dataset format!")

# Load validation dataset
print("\nLoading validation dataset...")
eval_dataset_raw = load_from_disk("/home/avramit/classifier/01122025/datasets/llmfinetune/mafat/parse/validation/")

# Apply eval dataset percentage filtering FIRST
if eval_dataset_percentage < 1.0:
    original_size = len(eval_dataset_raw)
    subset_size = int(original_size * eval_dataset_percentage)
    eval_dataset = eval_dataset_raw.select(range(subset_size))
    print(f"Using {eval_dataset_percentage*100}% of eval dataset: {subset_size}/{original_size} examples")
else:
    eval_dataset = eval_dataset_raw
    print(f"Using full eval dataset: {len(eval_dataset)} examples")

print(f"Eval dataset columns: {eval_dataset.column_names}")

# Apply formatting to eval dataset
if "text" not in eval_dataset.column_names and "instruction" in eval_dataset.column_names:
    print("📝 Formatting validation dataset...")
    
    def format_alpaca_simple_eval(examples):
        texts = []
        for instruction, input_val, output in zip(examples["instruction"], examples["input"], examples["output"]):
            text = alpaca_prompt.format(instruction, input_val, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    
    eval_dataset = eval_dataset.map(format_alpaca_simple_eval, batched=True, num_proc=1, remove_columns=["instruction", "input", "output"])
    print("   ✅ Eval formatting complete")
    print(f"   Final columns: {eval_dataset.column_names}")
elif "text" in eval_dataset.column_names:
    print("✅ Eval dataset already has 'text' field")
    # Debug: check first few samples
    print("   Checking first 5 samples for data integrity...")
    for i in range(min(5, len(eval_dataset))):
        sample_text = eval_dataset[i]["text"]
        print(f"   Sample {i}: type={type(sample_text).__name__}, len={len(sample_text) if isinstance(sample_text, str) else 'N/A'}, valid={isinstance(sample_text, str)}")
        if not isinstance(sample_text, str):
            print(f"      ⚠️ PROBLEM: This is {type(sample_text)}, not string!")
            print(f"      Content: {sample_text}")
    
    # Check if there are any non-string entries
    print("   Scanning all samples for non-string 'text' fields...")
    non_string_count = 0
    for i in range(len(eval_dataset)):
        if not isinstance(eval_dataset[i]["text"], str):
            non_string_count += 1
            if non_string_count <= 3:  # Show first 3 problematic samples
                print(f"      Sample {i} has invalid type: {type(eval_dataset[i]['text'])}")
    
    if non_string_count > 0:
        print(f"   ❌ Found {non_string_count} samples with non-string 'text' field!")
        print("   Filtering out invalid samples...")
        eval_dataset = eval_dataset.filter(lambda x: isinstance(x["text"], str))
        print(f"   ✅ Filtered dataset size: {len(eval_dataset)}")
    else:
        print(f"   ✅ All samples have valid string 'text' field")

# Print dataset sizes
print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(eval_dataset)}\n")

# Training configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,  # Reduced for long sequences
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Increased to compensate
        warmup_steps=50,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        max_grad_norm=1.0,
        dataset_num_proc=1,  # CRITICAL: Prevent 260 process spawn in tokenization
        dataloader_num_workers=0,  # Also disable dataloader multiprocessing
    ),
)

# Show memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train
print("Starting training...")
trainer_stats = trainer.train()

# Show final stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save model
print("Saving model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Training complete!")
