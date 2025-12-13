import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import Dataset
from tqdm import tqdm

print("v5 - GPU debug")
# ========================
# CONFIGURATION
# ========================
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity for additional context sentences
MAX_CONTEXT_SENTENCES = 3   # Maximum total context sentences (including original claim)
DUPLICATE_THRESHOLD = 0.95  # Threshold to detect duplicate sentences (avoid repetition)

# Prompt format options: "alpaca", "gemma", "chatml", or "multi"
PROMPT_FORMAT = "multi"  # Options: "alpaca", "gemma", "chatml", "multi"

# For "multi" mode - format distribution
MULTI_FORMAT_WEIGHTS = {
    "gemma": 0.50,   # Native Gemma format
    "alpaca": 0.35,  # Legacy instruction format
    "chatml": 0.15   # OpenAI/universal format
}

# Positive samples configuration
POSITIVE_SAMPLES_MODE = "all_untouched"  # Options: "none", "all_untouched", "ratio"
POSITIVE_RATIO = 2  # For "ratio" mode: for every N corruptions, create 1 positive sample

# Dataset size configuration
MAX_SAMPLES = None  # Maximum number of samples to parse from source (None = all samples)

# Load sentence transformer
print("Loading sentence transformer model...")
import torch
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Check and print device info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"🔢 GPU Count: {torch.cuda.device_count()}")
    embedder = embedder.to(device)
    print(f"✅ Sentence Transformer moved to GPU: {device}")
else:
    print(f"⚠️  Running on CPU - this will be VERY slow!")
print(f"📍 Embedder device: {embedder.device}")


def split_into_sentences(text):
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def extract_context_with_claim(article, original_claim, 
                                similarity_threshold=SIMILARITY_THRESHOLD,
                                max_sentences=MAX_CONTEXT_SENTENCES,
                                duplicate_threshold=DUPLICATE_THRESHOLD):
    """
    Extract context sentences: original claim + similar article sentences.
    
    Returns list of context sentences where:
    - First sentence is always the original claim
    - Following 1-2 sentences are most similar from article (if above threshold)
    - Total max is max_sentences (default 3)
    - Duplicates are filtered out
    """
    article_sentences = split_into_sentences(article)
    
    if len(article_sentences) == 0:
        return [original_claim]  # Return only original claim if no article sentences
    
    # Start with original claim as first context
    context = [original_claim]
    context_embeddings = [embedder.encode([original_claim])[0]]  # Store embeddings
    
    # Encode original claim and article sentences ONCE (this is the slow part!)
    claim_emb = embedder.encode([original_claim])
    sent_embs = embedder.encode(article_sentences)  # GPU encoding of all sentences
    
    # Compute similarities
    similarities = cosine_similarity(claim_emb, sent_embs)[0]
    
    # Sort by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Add article sentences that meet criteria
    for idx in sorted_indices:
        if len(context) >= max_sentences:
            break
            
        similarity = similarities[idx]
        candidate_sentence = article_sentences[idx]
        
        # Check if similarity is above threshold
        if similarity < similarity_threshold:
            break  # Stop if below threshold (sorted by similarity)
        
        # Check if sentence is not a duplicate of original claim
        if similarity >= duplicate_threshold:
            continue  # Skip near-duplicate sentences
        
        # Check if not duplicate of already added context (use pre-computed embeddings)
        is_duplicate = False
        candidate_emb = sent_embs[idx]  # Reuse pre-computed embedding
        for existing_emb in context_embeddings:
            sim = cosine_similarity([candidate_emb], [existing_emb])[0][0]
            if sim >= duplicate_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            context.append(candidate_sentence)
            context_embeddings.append(candidate_emb)  # Store for future comparisons
    
    return context


def format_alpaca_sample(instruction, input_text, output_json):
    """Format sample in Alpaca style"""
    # Create formatted text for compatibility with other formats
    output_str = json.dumps(output_json, ensure_ascii=False, indent=2)
    alpaca_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_str}"""
    
    return {
        "text": alpaca_text,
        "instruction": instruction,
        "input": input_text,
        "output": output_str
    }


def format_gemma_sample(context_sentences, transformed_claim, output_json):
    """Format sample in Gemma native style"""
    context_text = " ".join(context_sentences)
    
    text = f"""<bos><start_of_turn>user
You are a Hebrew factuality checker. Analyze the context and identify any factual errors in the corrupted claim.

CONTEXT: {context_text}

CORRUPTED_CLAIM: {transformed_claim}

Identify the error type and return as JSON.<end_of_turn>
<start_of_turn>model
{json.dumps(output_json, ensure_ascii=False, indent=2)}<end_of_turn><eos>"""
    
    return {"text": text}


def format_chatml_sample(context_sentences, transformed_claim, output_json):
    """Format sample in ChatML style (OpenAI standard)"""
    context_text = " ".join(context_sentences)
    
    text = f"""<|im_start|>system
You are a Hebrew factuality checker specialized in identifying errors in summarized claims.<|im_end|>
<|im_start|>user
CONTEXT: {context_text}

CORRUPTED_CLAIM: {transformed_claim}

Identify the factual error type and return as JSON.<|im_end|>
<|im_start|>assistant
{json.dumps(output_json, ensure_ascii=False, indent=2)}<|im_end|>"""
    
    return {"text": text}


def format_alpaca_positive_sample(instruction, input_text, output_json):
    """Format positive sample (no error) in Alpaca style"""
    # Create formatted text for compatibility with other formats
    output_str = json.dumps(output_json, ensure_ascii=False, indent=2)
    alpaca_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_str}"""
    
    return {
        "text": alpaca_text,
        "instruction": instruction,
        "input": input_text,
        "output": output_str
    }


def format_gemma_positive_sample(context_sentences, correct_claim, output_json):
    """Format positive sample (no error) in Gemma native style"""
    context_text = " ".join(context_sentences)
    
    text = f"""<bos><start_of_turn>user
You are a Hebrew factuality checker. Analyze the context and verify if the claim is factually correct.

CONTEXT: {context_text}

CLAIM: {correct_claim}

Identify if there are any errors and return as JSON.<end_of_turn>
<start_of_turn>model
{json.dumps(output_json, ensure_ascii=False, indent=2)}<end_of_turn><eos>"""
    
    return {"text": text}


def format_chatml_positive_sample(context_sentences, correct_claim, output_json):
    """Format positive sample (no error) in ChatML style"""
    context_text = " ".join(context_sentences)
    
    text = f"""<|im_start|>system
You are a Hebrew factuality checker specialized in identifying errors in summarized claims.<|im_end|>
<|im_start|>user
CONTEXT: {context_text}

CLAIM: {correct_claim}

Identify if there are any factual errors and return as JSON.<|im_end|>
<|im_start|>assistant
{json.dumps(output_json, ensure_ascii=False, indent=2)}<|im_end|>"""
    
    return {"text": text}


def create_training_samples(augmentation_json_path, output_path, use_evidence_window=True, prompt_format=PROMPT_FORMAT, 
                           positive_samples_mode=POSITIVE_SAMPLES_MODE, positive_ratio=POSITIVE_RATIO, max_samples=None):
    """Convert augmentation JSON to LLM training samples"""

    print(f"Loading augmentation data from {augmentation_json_path}...")
    with open(augmentation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Limit dataset size if specified
    if max_samples is not None and max_samples > 0:
        original_size = len(data)
        data = data[:max_samples]
        print(f"Limiting to {len(data)}/{original_size} samples (max_samples={max_samples})")

    samples = []
    total_corruptions = 0  # Track for ratio mode
    
    # Debug: Track timing
    import time
    total_encoding_time = 0
    total_context_extraction_time = 0
    num_context_calls = 0

    print("Processing samples...")
    for record in tqdm(data):
        source_article = record["source_article"]
        gold_summary = record.get("gold_summary", "")
        
        # Track which sentences from gold_summary are transformed
        transformed_sentences = set()

        for corrupted_obj in record.get("correctness_corrupted_summaries", []):
            transformation_list = corrupted_obj.get("transformation_list", [])

            for transform in transformation_list:
                if not transform.get("applied_transformations"):
                    continue  # Skip claims with no errors

                # Get error type (use first if multiple)
                error_type = transform["applied_transformations"][0]["type"]
                transformed_claim = transform["transformed_claim"]
                original_claim = transform["original_claim"]
                
                # Track this sentence as transformed
                transformed_sentences.add(original_claim.strip())

                # Extract context: original claim + similar article sentences
                start_time = time.time()
                context_sentences = extract_context_with_claim(
                    source_article,
                    original_claim,
                    similarity_threshold=SIMILARITY_THRESHOLD,
                    max_sentences=MAX_CONTEXT_SENTENCES,
                    duplicate_threshold=DUPLICATE_THRESHOLD
                )
                context_time = time.time() - start_time
                total_context_extraction_time += context_time
                num_context_calls += 1

                if not context_sentences:
                    continue  # Skip if no context found

                # Prepare input
                if use_evidence_window:
                    # Use context (original claim + article sentences)
                    article_context = " ".join(context_sentences)
                    input_text = f"CONTEXT:\n{article_context}\n\nCORRUPTED_CLAIM:\n{transformed_claim}"
                else:
                    # Use full article + original claim
                    input_text = f"ORIGINAL_CLAIM:\n{original_claim}\n\nARTICLE:\n{source_article}\n\nCORRUPTED_CLAIM:\n{transformed_claim}"

                # Create JSON output
                output_json = {
                    "error_type": error_type,
                    "corrupted_claim": transformed_claim,
                    "context": context_sentences  # Include all context
                }

                # Create samples based on prompt format
                instruction = 'Given context (original claim + article evidence) and a corrupted claim, identify the error type and return a JSON object: {"error_type": "...", "corrupted_claim": "...", "context": ["...", "..."]}'
                
                if prompt_format == "alpaca":
                    sample = format_alpaca_sample(instruction, input_text, output_json)
                    samples.append(sample)
                    
                elif prompt_format == "gemma":
                    sample = format_gemma_sample(context_sentences, transformed_claim, output_json)
                    samples.append(sample)
                    
                elif prompt_format == "chatml":
                    sample = format_chatml_sample(context_sentences, transformed_claim, output_json)
                    samples.append(sample)
                    
                elif prompt_format == "multi":
                    # Add all formats with weighted distribution
                    import random
                    format_choice = random.choices(
                        list(MULTI_FORMAT_WEIGHTS.keys()),
                        weights=list(MULTI_FORMAT_WEIGHTS.values()),
                        k=1
                    )[0]
                    
                    if format_choice == "alpaca":
                        sample = format_alpaca_sample(instruction, input_text, output_json)
                    elif format_choice == "gemma":
                        sample = format_gemma_sample(context_sentences, transformed_claim, output_json)
                    else:  # chatml
                        sample = format_chatml_sample(context_sentences, transformed_claim, output_json)
                    
                    samples.append(sample)
                    total_corruptions += 1
                    
                else:
                    raise ValueError(f"Invalid prompt_format: {prompt_format}. Must be 'alpaca', 'gemma', 'chatml', or 'multi'")
        
        # Generate positive samples based on mode
        if positive_samples_mode != "none" and gold_summary:
            # Split gold summary into sentences
            gold_sentences = split_into_sentences(gold_summary)
            
            # Find untouched sentences (not in transformation list)
            untouched_sentences = [s for s in gold_sentences if s.strip() not in transformed_sentences]
            
            # Determine how many positive samples to create
            if positive_samples_mode == "all_untouched":
                sentences_to_process = untouched_sentences
            elif positive_samples_mode == "ratio":
                # Calculate needed positive samples for this record's corruptions
                record_corruptions = len([s for s in samples if record["source_article"] in str(s)])
                needed = max(1, record_corruptions // positive_ratio) if record_corruptions > 0 else 0
                # Randomly select up to 'needed' sentences
                import random
                sentences_to_process = random.sample(untouched_sentences, min(needed, len(untouched_sentences))) if untouched_sentences else []
            else:
                sentences_to_process = []
            
            # Create positive samples
            for correct_claim in sentences_to_process:
                # Extract context for this correct claim
                start_time = time.time()
                context_sentences = extract_context_with_claim(
                    source_article,
                    correct_claim,
                    similarity_threshold=SIMILARITY_THRESHOLD,
                    max_sentences=MAX_CONTEXT_SENTENCES,
                    duplicate_threshold=DUPLICATE_THRESHOLD
                )
                context_time = time.time() - start_time
                total_context_extraction_time += context_time
                num_context_calls += 1
                
                if not context_sentences:
                    continue
                
                # Prepare input
                if use_evidence_window:
                    article_context = " ".join(context_sentences)
                    input_text = f"CONTEXT:\n{article_context}\n\nCLAIM:\n{correct_claim}"
                else:
                    input_text = f"CLAIM:\n{correct_claim}\n\nARTICLE:\n{source_article}"
                
                # Create JSON output for positive sample
                output_json = {
                    "error_type": "no_error",
                    "claim": correct_claim,
                    "context": context_sentences
                }
                
                # Create samples based on prompt format
                instruction = 'Given context and a claim, verify if the claim is factually correct and return a JSON object: {"error_type": "no_error", "claim": "...", "context": ["...", "..."]}'
                
                if prompt_format == "alpaca":
                    sample = format_alpaca_positive_sample(instruction, input_text, output_json)
                    samples.append(sample)
                    
                elif prompt_format == "gemma":
                    sample = format_gemma_positive_sample(context_sentences, correct_claim, output_json)
                    samples.append(sample)
                    
                elif prompt_format == "chatml":
                    sample = format_chatml_positive_sample(context_sentences, correct_claim, output_json)
                    samples.append(sample)
                    
                elif prompt_format == "multi":
                    import random
                    format_choice = random.choices(
                        list(MULTI_FORMAT_WEIGHTS.keys()),
                        weights=list(MULTI_FORMAT_WEIGHTS.values()),
                        k=1
                    )[0]
                    
                    if format_choice == "alpaca":
                        sample = format_alpaca_positive_sample(instruction, input_text, output_json)
                    elif format_choice == "gemma":
                        sample = format_gemma_positive_sample(context_sentences, correct_claim, output_json)
                    else:  # chatml
                        sample = format_chatml_positive_sample(context_sentences, correct_claim, output_json)
                    
                    samples.append(sample)

    print(f"\nCreated {len(samples)} training samples (format: {prompt_format})")
    print(f"  - Corrupted samples: {total_corruptions}")
    print(f"  - Positive samples: {len(samples) - total_corruptions}")
    print(f"  - Mode: {positive_samples_mode}")
    
    # Print timing statistics
    print(f"\n⏱️  Performance Statistics:")
    print(f"  - Total context extraction calls: {num_context_calls}")
    print(f"  - Total context extraction time: {total_context_extraction_time:.2f}s")
    if num_context_calls > 0:
        print(f"  - Average time per context extraction: {total_context_extraction_time/num_context_calls:.3f}s")
        print(f"  - Context extractions per second: {num_context_calls/total_context_extraction_time:.2f}")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(samples)

    # Save to disk
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")

    # Also save as JSON for inspection
    json_output_path = os.path.join(output_path, "samples_preview.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(samples[:10], f, ensure_ascii=False, indent=2)  # Save first 10 for inspection
    print(f"Sample preview saved to {json_output_path}")
    
    # Save configuration to text file
    import os
    from datetime import datetime
    config_path = os.path.join(output_path, "parse_config.txt")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PARSING CONFIGURATION\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source file: {augmentation_json_path}\n")
        f.write(f"Output path: {output_path}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("CONTEXT EXTRACTION SETTINGS\n")
        f.write("="*60 + "\n")
        f.write(f"SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}\n")
        f.write(f"MAX_CONTEXT_SENTENCES = {MAX_CONTEXT_SENTENCES}\n")
        f.write(f"DUPLICATE_THRESHOLD = {DUPLICATE_THRESHOLD}\n")
        f.write(f"use_evidence_window = {use_evidence_window}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("PROMPT FORMAT SETTINGS\n")
        f.write("="*60 + "\n")
        f.write(f"PROMPT_FORMAT = {prompt_format}\n")
        if prompt_format == "multi":
            f.write(f"MULTI_FORMAT_WEIGHTS = {MULTI_FORMAT_WEIGHTS}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("POSITIVE SAMPLES SETTINGS\n")
        f.write("="*60 + "\n")
        f.write(f"POSITIVE_SAMPLES_MODE = {positive_samples_mode}\n")
        f.write(f"POSITIVE_RATIO = {positive_ratio}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("DATASET SIZE SETTINGS\n")
        f.write("="*60 + "\n")
        f.write(f"MAX_SAMPLES = {max_samples if max_samples else 'None (all samples)'}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("DATASET STATISTICS\n")
        f.write("="*60 + "\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Corrupted samples: {total_corruptions}\n")
        f.write(f"Positive samples: {len(samples) - total_corruptions}\n")
        f.write("="*60 + "\n")
    print(f"Configuration saved to {config_path}")

    return dataset


if __name__ == "__main__":
    # Configuration
    AUGMENTATION_JSON = "/home/avramit/classifier/01122025/datasets/llmfinetune/mafat/train.json" # Your input file
    OUTPUT_DATASET = "/home/avramit/classifier/01122025/datasets/llmfinetune/mafat/parse/train/"  # Output directory

    # Create dataset
    dataset = create_training_samples(
        augmentation_json_path=AUGMENTATION_JSON,
        output_path=OUTPUT_DATASET,
        use_evidence_window=True,  # Set False to use full article
        prompt_format=PROMPT_FORMAT,  # "alpaca", "gemma", "chatml", or "multi"
        positive_samples_mode=POSITIVE_SAMPLES_MODE,  # "none", "all_untouched", or "ratio"
        positive_ratio=POSITIVE_RATIO,  # For "ratio" mode
        max_samples=MAX_SAMPLES  # Maximum samples to parse (None = all)
    )

    print("\nDataset statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print("\nFirst sample:")
    print(dataset[0])