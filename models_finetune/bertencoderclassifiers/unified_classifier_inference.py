"""
Unified inference script for both multi-label and binary classification models.
Supports two modes:
  1. Dataset Evaluation: Run inference on full CSV and compute metrics
  2. Single Sample: Inference + attention analysis for one text pair
"""

import logging
import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - USER EDITABLE
# ============================================================================

print("v4")
# Classification type: "multilabel" or "binary"
CLASSIFICATION_TYPE = "multilabel"  # Options: "multilabel", "binary"

# Inference mode: "dataset" or "single_sample"
INFERENCE_MODE = "single_sample"  # Options: "dataset", "single_sample"

# Model path
MODEL_PATH = "/home/avramit/classifier/binary/neodictabert-finetuned-4096-201125/"

# For dataset evaluation mode
TEST_FILE = "/home/avramit/classifier/datasets/dataset_nfacttrain_mafat/test.csv"
OUTPUT_DIR = "inference_output"
MAX_SEQ_LENGTH = 3072
BATCH_SIZE = 8
THRESHOLD = 0.5  # For multi-label classification

# For single sample mode
SAMPLE_PREMISE = """בשנת 1948 הוקמה מדינת ישראל לאחר מלחמת העצמאות."""
SAMPLE_HYPOTHESIS = """מדינת ישראל נוסדה בשנת 1947 על ידי דוד בן גוריון."""

# Attention visualization settings (only for single_sample mode)
ENABLE_ATTENTION_ANALYSIS = True  # Set to False if model doesn't support attention output
TOP_K_TOKENS = 15  # Show top K tokens by attention weight

# ============================================================================


@dataclass
class InferenceArguments:
    """Arguments for inference"""
    model_path: str = field(
        metadata={"help": "Path to the fine-tuned model directory"}
    )
    test_file: str = field(
        metadata={"help": "Path to test file (CSV format)"}
    )
    output_dir: str = field(
        default="inference_output",
        metadata={"help": "Directory to save predictions and metrics"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for inference"}
    )
    threshold: float = field(
        default=0.5,
        metadata={"help": "Probability threshold for multi-label classification"}
    )


def run_single_sample_inference(
    model_path: str,
    premise: str,
    hypothesis: str,
    classification_type: str,
    max_seq_length: int,
    threshold: float,
    top_k: int,
    enable_attention: bool = True
):
    """Run inference on single sample with attention analysis."""
    logger.info("=" * 60)
    logger.info("SINGLE SAMPLE")
    logger.info("=" * 60)

    # Load model with attention output enabled (if requested)
    logger.info(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # Get label information
    label_list = [config.id2label[i] for i in range(config.num_labels)]
    num_labels = len(label_list)

    logger.info(f"\nClassification Type: {classification_type.upper()}")
    logger.info(f"Number of Labels: {num_labels}")
    if classification_type == "binary":
        logger.info(f"Labels: {label_list}")
    else:
        logger.info(f"Error Types: {', '.join(label_list[:5])}... (showing first 5)")

    # Display input
    logger.info(f"\nPremise: {premise}")
    logger.info(f"Hypothesis: {hypothesis}")

    # Tokenize
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    )

    # Get tokens for attention visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference with attention capture
    captured_attentions = None
    if enable_attention:
        logger.info("\n" + "=" * 60)
        logger.info("CAPTURING ATTENTION")
        logger.info("=" * 60)

        # For NeoBERT, we need to pass output_attentions=True to the forward method
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,  # This tells NeoBERT to compute and return attention
            )

        # Check if attention was returned
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            captured_attentions = outputs.attentions
            logger.info(f"✅ Successfully captured {len(captured_attentions)} attention layers")
        else:
            logger.warning("⚠️  Model did not return attention weights")
    else:
        with torch.no_grad():
            outputs = model(**inputs)

    logits = outputs.logits[0].cpu()  # Shape: (num_labels,)

    # Check if attention outputs are available
    has_attention = captured_attentions is not None

    # Get predictions based on classification type
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    if classification_type == "binary":
        # Binary classification: use softmax
        probs = torch.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probs).item()
        predicted_class = label_list[predicted_class_idx]
        confidence = probs[predicted_class_idx].item()

        logger.info(f"\nPredicted: {predicted_class} ({confidence:.2%})")
        logger.info(f"\nProbabilities:")
        for i, label in enumerate(label_list):
            logger.info(f"  {label}: {probs[i].item():.4f}")

    else:
        # Multi-label classification: use sigmoid
        probs = torch.sigmoid(logits)
        predicted_indices = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()
        predicted_labels = [label_list[i] for i in predicted_indices]

        logger.info(f"\nPredicted ({len(predicted_labels)}):")
        if predicted_labels:
            for label_idx in predicted_indices:
                label = label_list[label_idx]
                prob = probs[label_idx].item()
                logger.info(f"  {label}: {prob:.4f}")
        else:
            logger.info("  None")

        logger.info(f"\nTop 10 probabilities:")
        sorted_indices = torch.argsort(probs, descending=True)[:10]
        for idx in sorted_indices:
            label = label_list[idx]
            prob = probs[idx].item()
            mark = "*" if idx in predicted_indices else " "
            logger.info(f"  {mark} {label}: {prob:.4f}")

    # Extract and analyze attention
    if not enable_attention:
        logger.info("\nAttention analysis disabled")
        return

    logger.info("\n" + "=" * 60)
    logger.info("ATTENTION ANALYSIS")
    logger.info("=" * 60)

    if not has_attention:
        logger.warning("⚠️  Attention not available from this model")
        logger.info("This may be due to:")
        logger.info("  1. Model using flash attention (disable by setting _attn_implementation='eager' in config)")
        logger.info("  2. Model architecture doesn't support attention output")
        return

    attentions = captured_attentions
    logger.info(f"Analyzing {len(attentions)} attention layers")

    last_layer_attention = attentions[-1]  # Shape: (batch=1, heads, seq_len, seq_len)

    # Average attention across all heads
    avg_attention = last_layer_attention.mean(dim=1)[0]  # Shape: (seq_len, seq_len)

    # Get attention from [CLS] token (used for classification)
    cls_attention = avg_attention[0, :]  # Shape: (seq_len,)

    # Find [SEP] token positions to identify premise vs hypothesis
    sep_indices = [i for i, token in enumerate(tokens) if token == '[SEP]']

    # Exclude special tokens for analysis
    content_start = 1  # Skip [CLS]
    content_end = len(tokens) - tokens[::-1].index('[PAD]') if '[PAD]' in tokens else len(tokens)

    # Normalize attention scores
    cls_attention_content = cls_attention[content_start:content_end]
    content_tokens = tokens[content_start:content_end]
    cls_attention_norm = cls_attention_content / cls_attention_content.sum()

    logger.info(f"\nLayer {len(attentions)}, {last_layer_attention.shape[1]} heads, [CLS] token attention")

    # Show premise vs hypothesis attention split
    if len(sep_indices) >= 1:
        premise_end_idx = sep_indices[0] - content_start
        if premise_end_idx > 0 and premise_end_idx < len(cls_attention_norm):
            premise_attention = cls_attention_norm[:premise_end_idx].sum().item()
            hypothesis_attention = cls_attention_norm[premise_end_idx:].sum().item()

            logger.info(f"\nAttention Distribution:")
            logger.info(f"  Premise:    {premise_attention:.1%}")
            logger.info(f"  Hypothesis: {hypothesis_attention:.1%}")

    # Show top K tokens by attention weight
    logger.info(f"\nTop {top_k} tokens by attention weight:")

    top_indices = torch.argsort(cls_attention_norm, descending=True)[:top_k]
    cumulative = 0.0
    for rank, idx in enumerate(top_indices, 1):
        token = content_tokens[idx]
        weight = cls_attention_norm[idx].item()
        cumulative += weight
        logger.info(f"{rank:2d}. {token:<25} {weight:.5f} (cumulative: {cumulative:.1%})")

    # Show full attention map (all tokens) - only if not too long
    if len(content_tokens) <= 50:
        logger.info(f"\nFull attention map ({len(content_tokens)} tokens):")
        for i, (token, weight) in enumerate(zip(content_tokens, cls_attention_norm)):
            bar_length = int(weight.item() * 100)
            bar = "█" * min(bar_length, 50)  # Cap bar at 50 chars
            logger.info(f"{i+1:3d}. {token:<20} {weight.item():.6f} {bar}")
    else:
        logger.info(f"\n(Skipping full attention map - {len(content_tokens)} tokens is too long)")


def run_dataset_evaluation(
    model_path: str,
    test_file: str,
    output_dir: str,
    classification_type: str,
    max_seq_length: int,
    batch_size: int,
    threshold: float
):
    """Run inference on full dataset and compute metrics."""
    logger.info("=" * 60)
    logger.info("DATASET EVALUATION")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model config to get label mappings
    logger.info(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Get label information
    label_list = [config.id2label[i] for i in range(config.num_labels)]
    label2id = config.label2id
    num_labels = len(label_list)

    logger.info(f"Classification Type: {classification_type.upper()}")
    logger.info(f"Detected {num_labels} labels: {label_list if classification_type == 'binary' else label_list[:5] + ['...']}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # Load test dataset
    logger.info(f"Loading test data from {test_file}...")
    extension = test_file.split(".")[-1]
    raw_dataset = load_dataset(extension, data_files={"test": test_file})["test"]
    logger.info(f"Loaded {len(raw_dataset)} samples")

    # Encode labels
    if classification_type == "multilabel":
        # Multi-label: encode error_types to multi-hot
        def encode_error_types(example):
            multi_hot = [0.0] * num_labels
            if example.get("error_types"):
                for tag in str(example["error_types"]).split(","):
                    tag = tag.strip()
                    if tag in label2id:
                        multi_hot[label2id[tag]] = 1.0
            example["labels"] = multi_hot
            return example

        logger.info("Encoding multi-hot labels...")
        test_dataset = raw_dataset.map(encode_error_types, batched=False)

    else:
        # Binary/multi-class: encode label to integer
        def encode_label(example):
            label_str = str(example.get("label", "")).strip()
            if label_str in label2id:
                example["labels"] = label2id[label_str]
            else:
                example["labels"] = 0  # Default
            return example

        logger.info("Encoding class labels...")
        test_dataset = raw_dataset.map(encode_label, batched=False)

    # Tokenize
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )

    logger.info("Tokenizing test dataset...")
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing test dataset"
    )

    # Prepare for inference
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # Run inference
    logger.info(f"Running inference on {len(test_dataset)} samples...")
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i + batch_size]

            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()
            labels = batch["labels"].numpy()

            all_logits.append(logits)
            all_labels.append(labels)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch['input_ids'])} / {len(test_dataset)} samples")

    # Combine all batches
    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels) if classification_type == "multilabel" else np.concatenate(all_labels)

    # Calculate metrics based on classification type
    logger.info("Calculating metrics...")

    if classification_type == "multilabel":
        # Multi-label metrics
        probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
        y_pred = (probs >= threshold).astype(int)
        y_true = all_labels.astype(int)

        # Overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )

        metrics = {
            "classification_type": "multilabel",
            "overall": {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "threshold": threshold,
                "num_samples": len(test_dataset)
            }
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

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-LABEL CLASSIFICATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total samples: {metrics['overall']['num_samples']}")
        logger.info(f"Overall Precision: {metrics['overall']['precision']:.4f}")
        logger.info(f"Overall Recall: {metrics['overall']['recall']:.4f}")
        logger.info(f"Overall F1: {metrics['overall']['f1']:.4f}")
        logger.info("\nPer-class metrics:")
        logger.info(f"{'Error Type':<40} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
        logger.info("-" * 100)
        for label in label_list:
            m = metrics["per_class"][label]
            logger.info(
                f"{label:<40} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                f"{m['f1']:<12.4f} {m['support']}"
            )

    else:
        # Binary/multi-class metrics
        probs = torch.softmax(torch.tensor(all_logits), dim=-1).numpy()
        y_pred = np.argmax(all_logits, axis=1)
        y_true = all_labels.astype(int)

        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        metrics = {
            "classification_type": "binary",
            "overall": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "num_samples": len(test_dataset)
            }
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

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("BINARY/MULTI-CLASS CLASSIFICATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total samples: {metrics['overall']['num_samples']}")
        logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
        logger.info(f"Overall Precision (macro): {metrics['overall']['precision']:.4f}")
        logger.info(f"Overall Recall (macro): {metrics['overall']['recall']:.4f}")
        logger.info(f"Overall F1 (macro): {metrics['overall']['f1']:.4f}")
        logger.info("\nPer-class metrics:")
        logger.info(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
        logger.info("-" * 70)
        for label in label_list:
            m = metrics["per_class"][label]
            logger.info(
                f"{label:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                f"{m['f1']:<12.4f} {m['support']}"
            )

    logger.info("=" * 80)

    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ Metrics saved to {metrics_file}")

    # Save predictions
    predictions_file = os.path.join(output_dir, "predictions.txt")
    with open(predictions_file, "w", encoding="utf-8") as f:
        f.write("index\t" + "\t".join(label_list) + "\n")
        for i, row in enumerate(probs):
            f.write(f"{i}\t" + "\t".join(f"{x:.4f}" for x in row) + "\n")
    logger.info(f"✅ Predictions saved to {predictions_file}")

    # Save detailed results
    results = []
    for i in range(min(len(test_dataset), 100)):  # Save first 100
        if classification_type == "multilabel":
            pred_labels = [label_list[j] for j in range(num_labels) if y_pred[i][j] == 1]
            true_labels = [label_list[j] for j in range(num_labels) if y_true[i][j] == 1]
        else:
            pred_labels = [label_list[y_pred[i]]]
            true_labels = [label_list[y_true[i]]]

        results.append({
            "index": i,
            "premise": raw_dataset[i]["premise"][:200] + "...",
            "hypothesis": raw_dataset[i]["hypothesis"],
            "true_labels": true_labels,
            "predicted_labels": pred_labels,
            "probabilities": {label_list[j]: float(probs[i][j]) for j in range(num_labels)}
        })

    results_file = os.path.join(output_dir, "detailed_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Detailed results (first 100) saved to {results_file}")

    logger.info(f"\n📁 All outputs saved to {output_dir}/")


def main():
    """Main entry point - routes to appropriate inference mode"""

    logger.info(f"\nClassification: {CLASSIFICATION_TYPE} | Mode: {INFERENCE_MODE}")
    logger.info(f"Model: {MODEL_PATH}")

    if INFERENCE_MODE == "single_sample":
        # Single sample inference with attention analysis
        run_single_sample_inference(
            model_path=MODEL_PATH,
            premise=SAMPLE_PREMISE,
            hypothesis=SAMPLE_HYPOTHESIS,
            classification_type=CLASSIFICATION_TYPE,
            max_seq_length=MAX_SEQ_LENGTH,
            threshold=THRESHOLD,
            top_k=TOP_K_TOKENS,
            enable_attention=ENABLE_ATTENTION_ANALYSIS
        )

    elif INFERENCE_MODE == "dataset":
        # Full dataset evaluation
        logger.info(f"Test: {TEST_FILE}")
        logger.info(f"Output: {OUTPUT_DIR}")

        run_dataset_evaluation(
            model_path=MODEL_PATH,
            test_file=TEST_FILE,
            output_dir=OUTPUT_DIR,
            classification_type=CLASSIFICATION_TYPE,
            max_seq_length=MAX_SEQ_LENGTH,
            batch_size=BATCH_SIZE,
            threshold=THRESHOLD
        )

    else:
        logger.error(f"Invalid INFERENCE_MODE: {INFERENCE_MODE}. Must be 'single_sample' or 'dataset'")
        return


if __name__ == "__main__":
    main()