"""
Finetuning multi-lingual models on error-type classification.
Adapted from Hugging Face's run_glue.py / run_xnli.py.
Now performs multi-label classification over summary-error tags,
with class-balanced loss and stratified oversampling.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import re
import datasets
import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.utils.data as data_utils

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry

print("check update v3")
# Ensure minimal versions
check_min_version("4.25.0.dev0")
require_version = getattr(transformers.utils.versions, "require_version", None)
if require_version:
    require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    max_seq_length: Optional[int] = field(default=128)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    train_dataset_percentage: Optional[float] = field(default=1.0, metadata={"help": "Percentage of training data to use (0.0-1.0)"})

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    language: Optional[str] = field(default=None)
    train_language: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    do_lower_case: Optional[bool] = field(default=False)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    send_example_telemetry("run_xnli_multilabel", model_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output dir ({training_args.output_dir}) exists and is not empty. "
                "Use --overwrite_output_dir to overwrite."
            )
        elif last_checkpoint:
            logger.info(f"Resuming from checkpoint {last_checkpoint}")

    set_seed(training_args.seed)

    # Print system info
    import socket
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Host IP: {socket.gethostbyname(socket.gethostname())}")
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Output dir: {training_args.output_dir}")
    print(f"Batch size: {training_args.per_device_train_batch_size} (effective: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps})")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Max seq length: {data_args.max_seq_length}")
    print(f"Train data %: {data_args.train_dataset_percentage*100}%")
    print("="*50)
    print("\nAll Arguments:")
    print(f"  Model: {vars(model_args)}")
    print(f"  Data: {vars(data_args)}")
    print(f"  Training: {vars(training_args)}")
    print("="*50 + "\n")

    data_files = {}
    extension = None
    if data_args.train_file:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    if not data_files:
        raise ValueError("Need to specify at least one of --train_file, --validation_file, --test_file")

    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # Data is already at claim level - no splitting needed
    
    def gather_error_types(ds):
        s = set()
        for ex in ds:
            val = ex.get("error_types")
            if isinstance(val, str) and val:
                for tag in val.split(","):
                    s.add(tag.strip())
        return s

    all_errors = set()
    for split in ("train", "validation", "test"):
        if split in raw_datasets:
            all_errors |= gather_error_types(raw_datasets[split])

    label_list = sorted(all_errors)
    label2id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    logger.info(f"Detected error types: {label_list}")

    def encode_error_types(example):
        multi_hot = [0.0] * num_labels
        if example.get("error_types"):
            for tag in example["error_types"].split(","):
                tag = tag.strip()
                if tag in label2id:
                    multi_hot[label2id[tag]] = 1.0
        example["labels"] = multi_hot
        return example

    # UPDATED: Remove all columns except premise, hypothesis, and labels
    for split in ("train", "validation", "test"):
        if split in raw_datasets:
            cols_to_remove = [col for col in raw_datasets[split].column_names 
                             if col not in ["premise", "hypothesis"] and col != "labels"]
            raw_datasets[split] = raw_datasets[split].map(
                encode_error_types,
                batched=False,
                remove_columns=cols_to_remove,
            )

    train_dataset = raw_datasets.get("train", None)
    eval_dataset = raw_datasets.get("validation", None)
    predict_dataset = raw_datasets.get("test", None)
    
    # Apply train dataset percentage filtering
    if train_dataset is not None and data_args.train_dataset_percentage < 1.0:
        original_size = len(train_dataset)
        subset_size = int(original_size * data_args.train_dataset_percentage)
        train_dataset = train_dataset.select(range(subset_size))
        logger.info(f"Using {data_args.train_dataset_percentage*100}% of train dataset: {subset_size}/{original_size} examples")
    
    # Print dataset sizes
    print(f"Dataset sizes - Train: {len(train_dataset) if train_dataset else 0}, "
          f"Val: {len(eval_dataset) if eval_dataset else 0}, "
          f"Test: {len(predict_dataset) if predict_dataset else 0}\n")

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={i: l for i, l in enumerate(label_list)},
        label2id=label2id,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        trust_remote_code=True,
    )

    padding = "max_length" if data_args.pad_to_max_length else False
    
    # Counter for exceeded samples
    exceeded_samples = {"count": 0, "split": ""}

    def preprocess_function(examples):
        lengths = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=False,
            truncation=False,
            return_length=True,
        )["length"]
        for i, length in enumerate(lengths):
            if length > data_args.max_seq_length:
                print(f"WARNING: Example {i} length {length} > max {data_args.max_seq_length}")
                exceeded_samples["count"] += 1
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

    if training_args.do_train and train_dataset is not None:
        if data_args.max_train_samples:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        exceeded_samples["count"] = 0
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing train dataset",
        ).remove_columns(["premise", "hypothesis"])
        if exceeded_samples["count"] > 0:
            print(f"⚠️  TRAIN: {exceeded_samples['count']} samples exceeded max_seq_length and were truncated\n")

    if training_args.do_eval and eval_dataset is not None:
        if data_args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
        exceeded_samples["count"] = 0
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing validation dataset",
        ).remove_columns(["premise", "hypothesis"])
        if exceeded_samples["count"] > 0:
            print(f"⚠️  VALIDATION: {exceeded_samples['count']} samples exceeded max_seq_length and were truncated\n")

    if training_args.do_predict and predict_dataset is not None:
        if data_args.max_predict_samples:
            predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))
        # Store original text and error types before tokenization
        test_texts = [{"premise": ex["premise"], "hypothesis": ex["hypothesis"], "error_types": ex.get("error_types", "")} for ex in predict_dataset]
        exceeded_samples["count"] = 0
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing test dataset",
        ).remove_columns(["premise", "hypothesis"])
        if exceeded_samples["count"] > 0:
            print(f"⚠️  TEST: {exceeded_samples['count']} samples exceeded max_seq_length and were truncated\n")

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Compute class-balanced pos_weight and stratified sampler
    all_labels_arr = np.stack(train_dataset["labels"])
    pos_counts = all_labels_arr.sum(axis=0)
    neg_counts = len(all_labels_arr) - pos_counts
    pos_weight = torch.tensor(
        [(neg_counts[i] / pos_counts[i]) if pos_counts[i] > 0 else 1.0
         for i in range(num_labels)],
        dtype=torch.float,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )

    example_weights = []
    for vec in all_labels_arr:
        pos_idxs = np.where(vec == 1)[0]
        if len(pos_idxs):
            w = np.mean(neg_counts[pos_idxs] / pos_counts[pos_idxs])
        else:
            w = np.mean(neg_counts / pos_counts)
        example_weights.append(w)

    sampler = data_utils.WeightedRandomSampler(
        weights=example_weights,
        num_samples=len(example_weights),
        replacement=True,
    )

    from sklearn.metrics import precision_recall_fscore_support

    def compute_metrics(p: EvalPrediction):
        logits = p.predictions if not isinstance(p.predictions, tuple) else p.predictions[0]
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        y_true = p.label_ids
        y_pred = (probs >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        return {"precision": prec, "recall": rec, "f1": f1}

    class WeightedTrainer(Trainer):
        def __init__(self, *args, pos_weight=None, sampler=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.sampler = sampler

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels").float()

            # Filter inputs to only include fields expected by the model
            model_inputs = {k: v for k, v in inputs.items()
                            if k in ['input_ids', 'attention_mask', 'token_type_ids']}

            outputs = model(**model_inputs)
            loss = self.loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):
            return data_utils.DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self.sampler,
                collate_fn=self.data_collator or default_data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
            )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset     if training_args.do_eval  else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        pos_weight=pos_weight,
        sampler=sampler,
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
        
        # Save script arguments
        import json
        args_file = os.path.join(training_args.output_dir, "script_arguments.json")
        with open(args_file, "w") as f:
            json.dump({
                "model_args": vars(model_args),
                "data_args": vars(data_args),
                "training_args": vars(training_args)
            }, f, indent=2, default=str)
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
    if training_args.do_predict:
        preds, labels, metrics = trainer.predict(predict_dataset)
        trainer.log_metrics("predict", metrics)
        output_file = os.path.join(training_args.output_dir, "predictions.txt")
        output_csv = os.path.join(training_args.output_dir, "predictions.csv")
        if trainer.is_world_process_zero():
            with open(output_file, "w") as writer:
                writer.write("index\t" + "\t".join(label_list) + "\n")
                probs = torch.sigmoid(torch.tensor(preds)).numpy()
                for i, row in enumerate(probs):
                    writer.write(f"{i}\t" + "\t".join(f"{x:.4f}" for x in row) + "\n")
            
            # Save CSV with text and scores
            import csv
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["premise", "hypothesis", "ground_truth_error_types"] + label_list
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                probs = torch.sigmoid(torch.tensor(preds)).numpy()
                for i, row in enumerate(probs):
                    row_data = {
                        "premise": test_texts[i]["premise"],
                        "hypothesis": test_texts[i]["hypothesis"],
                        "ground_truth_error_types": test_texts[i]["error_types"]
                    }
                    for j, label in enumerate(label_list):
                        row_data[label] = f"{row[j]:.4f}"
                    writer.writerow(row_data)

if __name__ == "__main__":
    main()