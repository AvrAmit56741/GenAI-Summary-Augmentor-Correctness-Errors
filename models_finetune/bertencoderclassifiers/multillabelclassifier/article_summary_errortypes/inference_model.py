"""
Inference script for multi-label error-type classification model.
Loads a fine-tuned model and runs predictions on a test file.
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
    HfArgumentParser,
)
from sklearn.metrics import precision_recall_fscore_support, classification_report

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceArguments:
    """Arguments for inference"""
    model_path: str = field(
        metadata={"help": "Path to the fine-tuned model directory"}
    )
    test_file: str = field(
        metadata={"help": "Path to test file (JSON format)"}
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
        metadata={"help": "Probability threshold for classification"}
    )


def split_summary_to_claims_batched(examples):
    """
    Split summaries into individual claims (same as training).
    """
    all_premises = []
    all_hypotheses = []
    all_error_types = []

    for premise, hypothesis, error_types in zip(
            examples["premise"],
            examples["hypothesis"],
            examples.get("error_types", [""] * len(examples["premise"]))
    ):
        if not hypothesis:
            all_premises.append(premise)
            all_hypotheses.append("")
            all_error_types.append(error_types)
            continue

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', hypothesis.strip())
        for sent in sentences:
            all_premises.append(premise)
            all_hypotheses.append(sent)
            all_error_types.append(error_types)

    return {
        "premise": all_premises,
        "hypothesis": all_hypotheses,
        "error_types": all_error_types
    }


def main():
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model config to get label mappings
    logger.info(f"Loading model from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    # Get label information
    label_list = [config.id2label[i] for i in range(config.num_labels)]
    label2id = config.label2id
    num_labels = len(label_list)

    logger.info(f"Detected {num_labels} error types: {label_list}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # Load test dataset
    logger.info(f"Loading test data from {args.test_file}...")
    extension = args.test_file.split(".")[-1]
    raw_dataset = load_dataset(extension, data_files={"test": args.test_file})["test"]

    # Split summaries into claims (same preprocessing as training)
    if "hypothesis" in raw_dataset.column_names:
        logger.info("Splitting summaries into claim-level samples...")
        raw_dataset = raw_dataset.map(
            split_summary_to_claims_batched,
            batched=True,
            batch_size=1000,
            remove_columns=raw_dataset.column_names,
            desc="Splitting summary into claims"
        )

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

    logger.info("Encoding labels...")
    test_dataset = raw_dataset.map(encode_error_types, batched=False)

    # Tokenize
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=args.max_seq_length,
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
        for i in range(0, len(test_dataset), args.batch_size):
            batch = test_dataset[i:i + args.batch_size]

            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()
            labels = batch["labels"].numpy()

            all_logits.append(logits)
            all_labels.append(labels)

            if (i // args.batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch['input_ids'])} / {len(test_dataset)} samples")

    # Combine all batches
    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)

    # Convert logits to probabilities
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()

    # Apply threshold
    y_pred = (probs >= args.threshold).astype(int)
    y_true = all_labels.astype(int)

    # Calculate metrics
    logger.info("Calculating metrics...")

    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    metrics = {
        "overall": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": args.threshold,
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

    # Save metrics
    metrics_file = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    # Save predictions
    predictions_file = os.path.join(args.output_dir, "predictions.txt")
    with open(predictions_file, "w", encoding="utf-8") as f:
        f.write("index\t" + "\t".join(label_list) + "\n")
        for i, row in enumerate(probs):
            f.write(f"{i}\t" + "\t".join(f"{x:.4f}" for x in row) + "\n")
    logger.info(f"Predictions saved to {predictions_file}")

    # Save detailed results
    results = []
    for i in range(len(test_dataset)):
        pred_labels = [label_list[j] for j in range(num_labels) if y_pred[i][j] == 1]
        true_labels = [label_list[j] for j in range(num_labels) if y_true[i][j] == 1]

        results.append({
            "index": i,
            "premise": raw_dataset[i]["premise"][:200] + "...",  # Truncate for readability
            "hypothesis": raw_dataset[i]["hypothesis"],
            "true_labels": true_labels,
            "predicted_labels": pred_labels,
            "probabilities": {label_list[j]: float(probs[i][j]) for j in range(num_labels)}
        })

    results_file = os.path.join(args.output_dir, "detailed_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results[:100], f, ensure_ascii=False, indent=2)  # Save first 100 for inspection
    logger.info(f"Detailed results (first 100) saved to {results_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total samples: {metrics['overall']['num_samples']}")
    logger.info(f"Overall Precision: {metrics['overall']['precision']:.4f}")
    logger.info(f"Overall Recall: {metrics['overall']['recall']:.4f}")
    logger.info(f"Overall F1: {metrics['overall']['f1']:.4f}")
    logger.info("\nPer-class metrics:")
    logger.info(f"{'Error Type':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
    logger.info("-" * 80)
    for label in label_list:
        m = metrics["per_class"][label]
        logger.info(
            f"{label:<30} {m['precision']:<12.4f} {m['recall']:<12.4f} "
            f"{m['f1']:<12.4f} {m['support']}"
        )
    logger.info("=" * 80)

    logger.info(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()