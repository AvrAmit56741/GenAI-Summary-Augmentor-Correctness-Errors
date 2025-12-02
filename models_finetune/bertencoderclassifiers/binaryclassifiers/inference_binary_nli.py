"""
Inference script for binary NLI classification (entailment/contradiction detection).
Simplified version with configuration inside the code.
"""

import json
import os
import csv
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
from collections import defaultdict

print ("update v1")
# ========================
# CONFIGURATION (Edit these values)
# ========================
MODEL_PATH = "/home/avramit/classifier/01122025/classifier/binary/neodictabert-finetuned-binary-011225/"
TEST_FILE = "/home/avramit/classifier/01122025/datasets/classifiers/test.csv"
OUTPUT_DIR = "/home/avramit/classifier/01122025/classifier/binary/neodictabert-finetuned-binary-011225/binary-inference_output"
MAX_SEQ_LENGTH = 4096  # Same as training
BATCH_SIZE = 8  # Adjust based on GPU memory (training used effective batch of 16)

# Test sample limit configuration
# Options:
#   - None = Test all samples
#   - N (integer) = Test only first N samples
MAX_TEST_SAMPLES = None

# Label filter configuration (optional)
# Options:
#   - None = Test all labels
#   - "entailment" or "contradiction" or "neutral" = Test only specific label
FILTER_LABEL = None

# Print each sample during inference (for debugging)
PRINT_EACH_SAMPLE = False


def load_model_and_tokenizer():
    """Load fine-tuned model and tokenizer"""
    import socket
    
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
    
    # Print system and model info
    print("\n" + "="*60)
    print("SYSTEM & MODEL INFORMATION")
    print("="*60)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU (⚠ WARNING: inference will be slow)")
    print(f"Host IP: {socket.gethostbyname(socket.gethostname())}")
    print(f"Model Type: {config.model_type}")
    print(f"Model Folder: {MODEL_PATH}")
    print("="*60 + "\n")
    
    # Get label information
    label_list = [config.id2label[i] for i in range(config.num_labels)]
    id2label = config.id2label
    label2id = config.label2id
    
    return model, tokenizer, device, label_list, id2label, label2id


def run_inference():
    """Main inference loop"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model, tokenizer, device, label_list, id2label, label2id = load_model_and_tokenizer()
    num_labels = len(label_list)
    
    print(f"Detected {num_labels} model labels: {label_list}")
    
    # Load test dataset
    print(f"\nLoading test dataset from {TEST_FILE}...")
    extension = TEST_FILE.split(".")[-1]
    raw_dataset = load_dataset(extension, data_files={"test": TEST_FILE})["test"]
    
    print(f"Original test dataset size: {len(raw_dataset)} samples")
    
    # Discover actual label names from the data
    actual_labels_in_data = set()
    for ex in raw_dataset:
        label_val = ex.get("label", "")
        if label_val:
            actual_labels_in_data.add(label_val)
    
    actual_label_list = sorted(list(actual_labels_in_data))
    print(f"Actual labels in dataset: {actual_label_list}")
    
    # Create mapping: actual label name -> model label index
    # Assume alphabetical order: first actual label -> 0, second -> 1
    actual_to_model_id = {label: i for i, label in enumerate(actual_label_list)}
    model_id_to_actual = {i: label for label, i in actual_to_model_id.items()}
    
    print(f"Label mapping: {actual_to_model_id}")
    
    # Filter by label if specified
    if FILTER_LABEL is not None:
        print(f"\nFiltering for label: {FILTER_LABEL}")
        raw_dataset = raw_dataset.filter(lambda x: x.get("label") == FILTER_LABEL)
        print(f"Filtered test dataset size: {len(raw_dataset)} samples")
    else:
        print("Testing on ALL labels")
    
    # Limit number of samples if specified
    if MAX_TEST_SAMPLES is not None and MAX_TEST_SAMPLES < len(raw_dataset):
        print(f"Limiting to first {MAX_TEST_SAMPLES} samples")
        raw_dataset = raw_dataset.select(range(MAX_TEST_SAMPLES))
        print(f"Limited test dataset size: {len(raw_dataset)} samples")
    
    # Store original text and labels (keeping string labels)
    test_texts = [
        {
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "label_str": ex.get("label", ""),  # Keep original string label
            "label": actual_to_model_id.get(ex.get("label", ""), 0)  # Convert to model index
        }
        for ex in raw_dataset
    ]
    
    # Count label distribution (using actual label names)
    label_counts = defaultdict(int)
    for item in test_texts:
        label_counts[item["label_str"]] += 1
    
    print("\n" + "="*60)
    print("DATASET LABEL DISTRIBUTION")
    print("="*60)
    for label_name in sorted(label_counts.keys()):
        print(f"{label_name}: {label_counts[label_name]} samples")
    print("="*60 + "\n")
    
    # Tokenize
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        )
    
    print("\nTokenizing test dataset...")
    test_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing"
    )
    
    # Prepare for inference
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )
    
    # Run inference
    print(f"\nRunning inference on {len(test_dataset)} samples...\n")
    all_logits = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), BATCH_SIZE)):
            batch = test_dataset[i:i + BATCH_SIZE]
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)
            
            # Print samples if enabled
            if PRINT_EACH_SAMPLE:
                batch_size = len(batch["input_ids"])
                for j in range(batch_size):
                    idx = i + j
                    probs = torch.softmax(torch.tensor(logits[j]), dim=0).numpy()
                    pred_label_idx = np.argmax(probs)
                    pred_label = id2label[pred_label_idx]
                    true_label = test_texts[idx]["label"]
                    true_label_str = id2label[true_label] if isinstance(true_label, int) else true_label
                    
                    print(f"\n--- Sample {idx} ---")
                    print(f"Premise: {test_texts[idx]['premise'][:100]}...")
                    print(f"Hypothesis: {test_texts[idx]['hypothesis']}")
                    print(f"True label: {true_label_str}")
                    print(f"Predicted: {pred_label} (confidence: {probs[pred_label_idx]:.4f})")
                    print(f"Probabilities: {dict(zip(label_list, [f'{p:.4f}' for p in probs]))}")
    
    # Combine all batches
    all_logits = np.vstack(all_logits)
    
    # Convert logits to probabilities
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    
    # Get predictions
    y_pred = np.argmax(probs, axis=1)
    y_true = np.array([test_texts[i]["label"] for i in range(len(test_texts))])
    
    # Convert predictions and true labels to actual label names
    y_pred_labels = [model_id_to_actual[int(pred)] for pred in y_pred]
    y_true_labels = [test_texts[i]["label_str"] for i in range(len(test_texts))]
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(num_labels)
    )
    
    # Macro and micro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_labels))
    
    metrics = {
        "overall": {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_micro": float(precision_micro),
            "recall_micro": float(recall_micro),
            "f1_micro": float(f1_micro),
            "num_samples": len(test_dataset)
        },
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "label_distribution": defaultdict(int)
    }
    
    # Per-class metrics (use actual label names)
    for i, label in enumerate(actual_label_list):
        metrics["per_class"][label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
    
    # Label distribution (already calculated above, just copy)
    metrics["label_distribution"] = dict(label_counts)
    
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
    
    # Save predictions CSV with probabilities for each class
    predictions_csv = os.path.join(OUTPUT_DIR, "predictions.csv")
    with open(predictions_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["premise", "hypothesis", "true_label", "predicted_label"] + [f"prob_{label}" for label in actual_label_list]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(test_dataset)):
            row_data = {
                "premise": test_texts[i]["premise"],
                "hypothesis": test_texts[i]["hypothesis"],
                "true_label": y_true_labels[i],
                "predicted_label": y_pred_labels[i]
            }
            for j, label in enumerate(actual_label_list):
                row_data[f"prob_{label}"] = f"{probs[i][j]:.4f}"
            writer.writerow(row_data)
    
    print(f"Predictions CSV saved to {predictions_csv}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Total samples: {metrics['overall']['num_samples']}")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"\nMacro averages:")
    print(f"  Precision: {metrics['overall']['precision_macro']:.4f}")
    print(f"  Recall: {metrics['overall']['recall_macro']:.4f}")
    print(f"  F1: {metrics['overall']['f1_macro']:.4f}")
    print(f"\nMicro averages:")
    print(f"  Precision: {metrics['overall']['precision_micro']:.4f}")
    print(f"  Recall: {metrics['overall']['recall_micro']:.4f}")
    print(f"  F1: {metrics['overall']['f1_micro']:.4f}")
    
    print("\nPer-class metrics:")
    print(f"{'Label':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
    print("-" * 80)
    for label in actual_label_list:
        m = metrics["per_class"][label]
        print(f"{label:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']}")
    
    print("\nLabel distribution:")
    for label in sorted(metrics["label_distribution"].keys()):
        count = metrics["label_distribution"][label]
        print(f"  {label}: {count}")
    
    print("\nConfusion Matrix:")
    print(f"{'':>15}", end="")
    for label in actual_label_list:
        print(f"{label[:10]:>12}", end="")
    print()
    for i, label in enumerate(actual_label_list):
        print(f"{label[:15]:>15}", end="")
        for j in range(len(actual_label_list)):
            print(f"{cm[i][j]:>12}", end="")
        print()
    
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
    print(f"Filter label: {FILTER_LABEL if FILTER_LABEL else 'None (all labels)'}")
    print(f"Max test samples: {MAX_TEST_SAMPLES if MAX_TEST_SAMPLES else 'None (all samples)'}")
    print(f"Print each sample: {PRINT_EACH_SAMPLE}")
    print("="*80 + "\n")
    
    metrics = run_inference()
    print("\nInference complete!")

