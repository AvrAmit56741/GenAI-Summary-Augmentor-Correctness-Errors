import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import Dataset
from tqdm import tqdm

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
    "alpaca": 0.25,  # Legacy instruction format
    "chatml": 0.25   # OpenAI/universal format
}

# Load sentence transformer
print("Loading sentence transformer model...")
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


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
    
    # Encode original claim and article sentences
    claim_emb = embedder.encode([original_claim])
    sent_embs = embedder.encode(article_sentences)
    
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
        
        # Check if not duplicate of already added context
        is_duplicate = False
        candidate_emb = embedder.encode([candidate_sentence])
        for existing_sentence in context:
            existing_emb = embedder.encode([existing_sentence])
            sim = cosine_similarity(candidate_emb, existing_emb)[0][0]
            if sim >= duplicate_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            context.append(candidate_sentence)
    
    return context


def format_alpaca_sample(instruction, input_text, output_json):
    """Format sample in Alpaca style"""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": json.dumps(output_json, ensure_ascii=False, indent=2)
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


def create_training_samples(augmentation_json_path, output_path, use_evidence_window=True, prompt_format=PROMPT_FORMAT):
    """Convert augmentation JSON to LLM training samples"""

    print(f"Loading augmentation data from {augmentation_json_path}...")
    with open(augmentation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []

    print("Processing samples...")
    for record in tqdm(data):
        source_article = record["source_article"]

        for corrupted_obj in record.get("correctness_corrupted_summaries", []):
            transformation_list = corrupted_obj.get("transformation_list", [])

            for transform in transformation_list:
                if not transform.get("applied_transformations"):
                    continue  # Skip claims with no errors

                # Get error type (use first if multiple)
                error_type = transform["applied_transformations"][0]["type"]
                transformed_claim = transform["transformed_claim"]
                original_claim = transform["original_claim"]

                # Extract context: original claim + similar article sentences
                context_sentences = extract_context_with_claim(
                    source_article,
                    original_claim,
                    similarity_threshold=SIMILARITY_THRESHOLD,
                    max_sentences=MAX_CONTEXT_SENTENCES,
                    duplicate_threshold=DUPLICATE_THRESHOLD
                )

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
                    
                else:
                    raise ValueError(f"Invalid prompt_format: {prompt_format}. Must be 'alpaca', 'gemma', 'chatml', or 'multi'")

    print(f"\nCreated {len(samples)} training samples (format: {prompt_format})")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(samples)

    # Save to disk
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")

    # Also save as JSON for inspection
    json_output_path = output_path.replace('.dataset', '.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(samples[:10], f, ensure_ascii=False, indent=2)  # Save first 10 for inspection
    print(f"Sample preview saved to {json_output_path}")

    return dataset


if __name__ == "__main__":
    # Configuration
    AUGMENTATION_JSON = "/home/avramit/llm_finetune/llm_evidence_big_dataset/train.json" # Your input file
    OUTPUT_DATASET = "/home/avramit/llm_finetune/llm_finetune_big_parse_dataset/llm_evidence_dataset_train"  # Output directory

    # Create dataset
    dataset = create_training_samples(
        augmentation_json_path=AUGMENTATION_JSON,
        output_path=OUTPUT_DATASET,
        use_evidence_window=True,  # Set False to use full article
        prompt_format=PROMPT_FORMAT  # "alpaca", "gemma", or "mix"
    )

    print("\nDataset statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print("\nFirst sample:")
    print(dataset[0])