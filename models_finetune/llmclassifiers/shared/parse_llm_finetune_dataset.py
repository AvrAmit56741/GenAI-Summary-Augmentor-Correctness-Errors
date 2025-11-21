import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import Dataset
from tqdm import tqdm

# Load sentence transformer
print("Loading sentence transformer model...")
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def split_into_sentences(text):
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def extract_top_k_evidence(article, original_claim, k=3):
    """Extract top-K most similar sentences from article"""
    article_sentences = split_into_sentences(article)

    if len(article_sentences) == 0:
        return []

    if len(article_sentences) <= k:
        return article_sentences

    # Encode claim and sentences
    claim_emb = embedder.encode([original_claim])
    sent_embs = embedder.encode(article_sentences)

    # Compute similarities
    similarities = cosine_similarity(claim_emb, sent_embs)[0]

    # Get top-K indices
    top_k_idx = np.argsort(similarities)[-k:][::-1]

    return [article_sentences[i] for i in top_k_idx]


def create_training_samples(augmentation_json_path, output_path, use_evidence_window=True):
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

                # Extract evidence sentences using original_claim
                evidence_sentences = extract_top_k_evidence(
                    source_article,
                    original_claim,
                    k=3
                )

                if not evidence_sentences:
                    continue  # Skip if no evidence found

                # Prepare input
                if use_evidence_window:
                    # Use only top-3 evidence sentences as context
                    article_context = " ".join(evidence_sentences)
                    input_text = f"ARTICLE:\n{article_context}\n\nSUMMARY_CLAIM:\n{transformed_claim}"
                else:
                    # Use full article
                    input_text = f"ARTICLE:\n{source_article}\n\nSUMMARY_CLAIM:\n{transformed_claim}"

                # Create JSON output
                output_json = {
                    "error_type": error_type,
                    "summary_claim": transformed_claim,
                    "article_evidence": evidence_sentences[:1]  # Top-1 for output
                }

                sample = {
                    "instruction": 'Given a Hebrew article and ONE corrupted claim from its summary, identify the error type and return a JSON object: {"error_type": "...", "summary_claim": "...", "article_evidence": ["...", "..."]}',
                    "input": input_text,
                    "output": json.dumps(output_json, ensure_ascii=False, indent=2)
                }
                samples.append(sample)

    print(f"\nCreated {len(samples)} training samples")

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
        use_evidence_window=True  # Set False to use full article
    )

    print("\nDataset statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print("\nFirst sample:")
    print(dataset[0])