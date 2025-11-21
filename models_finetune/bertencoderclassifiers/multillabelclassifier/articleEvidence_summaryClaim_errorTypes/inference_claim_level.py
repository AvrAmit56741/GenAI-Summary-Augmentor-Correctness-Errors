"""
Inference script for multi-label error-type classification model (BERT-based).
Simplified version with configuration inside the code.
"""

import json
import os
import csv
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from collections import defaultdict

print("update 15")
# Error type constants
class ErrorTypes:
    # Morphological & Syntactic
    IDIOM_COLLOCATION_CORRUPTION = "idiom_collocation_corruption"
    MORPHOLOGICAL_CONNECTIVE_CONFUSION = "morphological_connective_confusion"
    NOUN_GENDER_SWAP = "noun_gender_swap"
    VERB_GENDER_SWAP = "verb_gender_swap"
    VERB_TENSE_SWAP = "verb_tense_swap"
    DIRECTIONAL_PREPOSITION_SWAP = "directional_preposition_swap"
    
    # Linguistic Phenomena
    PRONOUN_REFERENCE_ERRORS = "pronoun_reference_errors"
    SPECIFICITY_SHIFT_ERRORS = "specificity_shift_errors"
    HOMOGRAPHIC_GENDER_ERRORS = "homographic_gender_errors"
    HEBREW_ROOT_PATTEN_CONFUSION = "hebrew_root_pattern_confusion"
    IMPERSONAL_TO_PERSONAL_VERB_ERRORS = "impersonal_to_personal_verb_errors"
    INCREMENTAL_TEMPORAL_STATE_MUTATIONS = "incremental_temporal_state_mutations"
    EVIDENTIALITY_SOURCE_ATTRIBUTION_COLLAPSE = "evidentiality_source_attribution_collapse"
    VERB_TEMPLATE_AGENCY_REVERSAL = "verb_template_agency_reversal"
    DOUBLE_VERB_DEPENDENCY_ERRORS = "double_verb_dependency_errors"
    CONSTRUCT_STATE_CONFUSION = "construct_state_confusion"
    SUBJECT_VERB_PERSON_MISMATCH = "subject_verb_person_mismatch"
    
    # Misc
    MEASURE_UNIT_SWAP = "measure_unit_swap"
    NUMBER_SWAP = "number_swap"
    SENTENCE_NEGATION = "sentence_negation"
    
    # Entity Swaps
    ENTITY_TITLE_SWAP = "entity_title_swap"
    ENTITY_EVENT_SWAP = "entity_event_swap"
    ENTITY_LOCATION_SWAP = "entity_location_swap"
    ENTITY_PERSON_SWAP = "entity_person_swap"
    ENTITY_ORGANIZATION_SWAP = "entity_organization_swap"
    ENTITY_DATE_SWAP = "entity_date_swap"

# ========================
# CONFIGURATION (Edit these values)
# ========================
MODEL_PATH = "/home/avramit/classifier/LongHeRo-claimlevel-finetuned-multilabel_error_types_191125_v2/"
TEST_FILE = "/home/avramit/llm_finetune/llm_evidence_big_dataset/parse_orgclaim_augclaim/test.csv"
OUTPUT_DIR = "/home/avramit/classifier/LongHeRo-claimlevel-finetuned-multilabel_error_types_191125_v2/inference_output"
MAX_SEQ_LENGTH = 3072
BATCH_SIZE = 8
THRESHOLD = 0.5  # Probability threshold for positive classification

# Test filtering configuration
# Options: 
#   - None or [] = Test all samples
#   - [ErrorTypes.NUMBER_SWAP, ErrorTypes.SENTENCE_NEGATION] = Test only specific error types
FILTER_ERROR_TYPES = None

# Test sample limit configuration
# Options:
#   - None = Test all samples (after filtering)
#   - N (integer) = Test only first N samples (after filtering)
MAX_TEST_SAMPLES = 10


def load_model_and_tokenizer():
    """Load fine-tuned model and tokenizer"""
    print("Loading model and tokenizer...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Print device info
    if device.type == "cuda":
        print(f"✓ Model running on CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ WARNING: Model running on CPU (inference will be slow)")
    
    # Get label information
    label_list = [config.id2label[i] for i in range(config.num_labels)]
    label2id = config.label2id
    
    return model, tokenizer, device, label_list, label2id


def filter_dataset_by_error_types(dataset, error_types):
    """Filter dataset to only include samples with specific error types"""
    if not error_types:
        return dataset
    
    def filter_fn(example):
        if not example.get("error_types"):
            return False
        sample_error_types = [et.strip() for et in example["error_types"].split(",")]
        return any(et in error_types for et in sample_error_types)
    
    return dataset.filter(filter_fn)


def run_inference():
    """Main inference loop"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model, tokenizer, device, label_list, label2id = load_model_and_tokenizer()
    num_labels = len(label_list)
    
    print(f"Detected {num_labels} error types: {label_list[:5]}... (showing first 5)")
    
    # Load test dataset
    print(f"\nLoading test dataset from {TEST_FILE}...")
    extension = TEST_FILE.split(".")[-1]
    raw_dataset = load_dataset(extension, data_files={"test": TEST_FILE})["test"]
    
    print(f"Original test dataset size: {len(raw_dataset)} samples")
    
    # Filter by error types if specified
    if FILTER_ERROR_TYPES:
        print(f"\nFiltering for error types: {FILTER_ERROR_TYPES}")
        raw_dataset = filter_dataset_by_error_types(raw_dataset, FILTER_ERROR_TYPES)
        print(f"Filtered test dataset size: {len(raw_dataset)} samples")
    else:
        print("Testing on ALL error types")
    
    # Limit number of samples if specified
    if MAX_TEST_SAMPLES is not None and MAX_TEST_SAMPLES < len(raw_dataset):
        print(f"Limiting to first {MAX_TEST_SAMPLES} samples")
        raw_dataset = raw_dataset.select(range(MAX_TEST_SAMPLES))
        print(f"Limited test dataset size: {len(raw_dataset)} samples")
    
    # Encode error types to multi-hot labels
    def encode_error_types(example):
        multi_hot = [0.0] * num_labels
        if example.get("error_types"):
            for tag in example["error_types"].split(","):
                tag = tag.strip()
                if tag in label2id:
                    multi_hot[label2id[tag]] = 1.0
        example["labels"] = multi_hot
        return example
    
    print("\nEncoding labels...")
    test_dataset = raw_dataset.map(encode_error_types, batched=False)
    
    # Store original text before tokenization
    test_texts = [
        {
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "error_types": ex.get("error_types", "")
        }
        for ex in test_dataset
    ]
    
    # Tokenize
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        )
    
    print("Tokenizing test dataset...")
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing"
    )
    
    # Prepare for inference
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    # Run inference
    print(f"\nRunning inference on {len(test_dataset)} samples...\n")
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), BATCH_SIZE)):
            batch = test_dataset[i:i + BATCH_SIZE]
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()
            labels = batch["labels"].numpy()
            
            all_logits.append(logits)
            all_labels.append(labels)
    
    # Combine all batches
    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)
    
    # Convert logits to probabilities
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    
    # Apply threshold
    y_pred = (probs >= THRESHOLD).astype(int)
    y_true = all_labels.astype(int)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    
    metrics = {
        "overall": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": THRESHOLD,
            "num_samples": len(test_dataset)
        },
        "error_type_distribution": defaultdict(int),
        "error_type_correct": defaultdict(int)
    }
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics["per_class"] = {}
    for i, label in enumerate(label_list):
        metrics["per_class"][label] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i])
        }
    
    # Count error type distribution and correct predictions
    for i in range(len(test_dataset)):
        true_error_types = test_texts[i]["error_types"]
        if true_error_types:
            for et in true_error_types.split(","):
                et = et.strip()
                metrics["error_type_distribution"][et] += 1
                
                # Check if predicted correctly
                if et in label2id:
                    label_idx = label2id[et]
                    if y_pred[i][label_idx] == 1:
                        metrics["error_type_correct"][et] += 1
    
    # Save metrics
    metrics_file = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        # Convert defaultdict to dict for JSON serialization
        metrics_serializable = {
            k: dict(v) if isinstance(v, defaultdict) else v
            for k, v in metrics.items()
        }
        json.dump(metrics_serializable, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    # Save predictions CSV
    predictions_csv = os.path.join(OUTPUT_DIR, "predictions.csv")
    with open(predictions_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["premise", "hypothesis", "ground_truth_error_types"] + label_list
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(test_dataset)):
            row_data = {
                "premise": test_texts[i]["premise"],
                "hypothesis": test_texts[i]["hypothesis"],
                "ground_truth_error_types": test_texts[i]["error_types"]
            }
            for j, label in enumerate(label_list):
                row_data[label] = f"{probs[i][j]:.4f}"
            writer.writerow(row_data)
    
    print(f"Predictions CSV saved to {predictions_csv}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Total samples: {metrics['overall']['num_samples']}")
    print(f"Overall Precision: {metrics['overall']['precision']:.4f}")
    print(f"Overall Recall: {metrics['overall']['recall']:.4f}")
    print(f"Overall F1: {metrics['overall']['f1']:.4f}")
    print(f"Threshold: {THRESHOLD}")
    print("\nTop error types by support:")
    
    # Sort by support and show top 10
    per_class_sorted = sorted(
        metrics["per_class"].items(),
        key=lambda x: x[1]["support"],
        reverse=True
    )[:20]
    
    print(f"{'Error Type':<40} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
    print("-" * 80)
    for label, m in per_class_sorted:
        print(f"{label:<40} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']}")
    print("=" * 80)
    
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    
    return metrics


if __name__ == "__main__":
    # Print configuration
    print("\n" + "="*80)
    print("INFERENCE CONFIGURATION")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Test file: {TEST_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max seq length: {MAX_SEQ_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Filter error types: {FILTER_ERROR_TYPES if FILTER_ERROR_TYPES else 'None (all types)'}")
    print(f"Max test samples: {MAX_TEST_SAMPLES if MAX_TEST_SAMPLES else 'None (all samples)'}")
    print("="*80 + "\n")
    
    metrics = run_inference()
    print("\nInference complete!")

