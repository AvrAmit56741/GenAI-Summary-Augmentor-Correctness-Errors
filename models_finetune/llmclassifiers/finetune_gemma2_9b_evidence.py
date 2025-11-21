from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer

print ("update v1")
# Model configuration
max_seq_length = 4096  # Increased for article + claim
dtype = None
load_in_4bit = True

# Dataset configuration
dataset_percentage = 0.85 # Use 100% of dataset (set to 0.8 for 80%, 0.7 for 70%, etc.)
eval_dataset_percentage = 0.2 # Use 100% of eval dataset

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
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Print system info
import socket
output_dir = "/home/avramit/llm_finetune/hebrew_evidence_llm_lora_100p"
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
train_dataset = load_from_disk("/home/avramit/llm_finetune/llm_finetune_big_parse_dataset/llm_evidence_dataset_train/")

# Apply dataset percentage filtering
if dataset_percentage < 1.0:
    original_size = len(train_dataset)
    subset_size = int(original_size * dataset_percentage)
    train_dataset = train_dataset.select(range(subset_size))
    print(f"Using {dataset_percentage*100}% of train dataset: {subset_size}/{original_size} examples")
else:
    print(f"Using full train dataset: {len(train_dataset)} examples")

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

# Load validation dataset
print("Loading validation dataset...")
eval_dataset = load_from_disk("/home/avramit/llm_finetune/llm_finetune_big_parse_dataset/llm_evidence_dataset_val/")
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

# Apply eval dataset percentage filtering
if eval_dataset_percentage < 1.0:
    original_size = len(eval_dataset)
    subset_size = int(original_size * eval_dataset_percentage)
    eval_dataset = eval_dataset.select(range(subset_size))
    print(f"Using {eval_dataset_percentage*100}% of eval dataset: {subset_size}/{original_size} examples")
else:
    print(f"Using full eval dataset: {len(eval_dataset)} examples")

# Print dataset sizes
print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(eval_dataset)}\n")

# Training configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    dataset_num_proc=1,  # Disable multiprocessing to avoid crashes
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
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        max_grad_norm=1.0,
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