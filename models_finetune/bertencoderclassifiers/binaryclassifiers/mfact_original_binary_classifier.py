#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset
import torch

import evaluate
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
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    get_predict_scores: bool = field(
        default=False, metadata={"help": "Get predicted scores rather than labels."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    language: str = field(
        default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    )
    train_language: Optional[str] = field(
        default=None, metadata={"help": "Train language if it is different from the evaluation language."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    print("hello")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_xnli", model_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    if (data_args.train_file is not None) or (data_args.validation_file is not None) or (data_args.test_file is not None):
        data_files = {}
        if training_args.do_train and data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if training_args.do_eval and data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if training_args.do_predict and data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            #use_auth_token=True if model_args.use_auth_token else None,
        )
        # Convert label column to labelFeature in HF
        raw_datasets = raw_datasets.class_encode_column("label")
        if training_args.do_train:
            train_dataset=raw_datasets['train']
            label_list = train_dataset.features["label"].names
        if training_args.do_eval:
            eval_dataset=raw_datasets['validation']
            label_list = eval_dataset.features["label"].names
        if training_args.do_predict:
            predict_dataset=raw_datasets['test']
            label_list = predict_dataset.features["label"].names

    else:
        if training_args.do_train:
            if model_args.train_language is None:
                train_dataset = load_dataset(
                    "xnli",
                    model_args.language,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
            else:
                train_dataset = load_dataset(
                    "xnli",
                    model_args.train_language,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
                label_list = train_dataset.features["label"].names

        if training_args.do_eval:
            eval_dataset = load_dataset(
                "xnli",
                model_args.language,
                split="validation",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            label_list = eval_dataset.features["label"].names

        if training_args.do_predict:
            predict_dataset = load_dataset(
                "xnli",
                model_args.language,
                split="test",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            label_list = predict_dataset.features["label"].names

    # Labels
    print("The label list is",label_list)
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Print system info and save to file
    import socket
    import json as json_lib
    from datetime import datetime
    
    training_info = []
    
    def log_and_save(msg):
        print(msg)
        training_info.append(msg)
    
    log_and_save("\n" + "="*50)
    log_and_save("TRAINING CONFIGURATION")
    log_and_save("="*50)
    if torch.cuda.is_available():
        log_and_save(f"GPU: {torch.cuda.get_device_name(0)}")
    log_and_save(f"Host IP: {socket.gethostbyname(socket.gethostname())}")
    log_and_save(f"Model: {model_args.model_name_or_path}")
    log_and_save(f"Output dir: {training_args.output_dir}")
    log_and_save(f"Batch size: {training_args.per_device_train_batch_size}")
    log_and_save(f"Learning rate: {training_args.learning_rate}")
    log_and_save(f"Epochs: {training_args.num_train_epochs}")
    log_and_save(f"Max seq length: {data_args.max_seq_length}")
    log_and_save(f"Labels: {label_list}")
    log_and_save("="*50 + "\n")
    
    # Save all arguments to JSON
    all_args = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "training_args": training_args.to_dict()
    }
    
    args_file = os.path.join(training_args.output_dir, "training_info_args.json")
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(args_file, "w", encoding="utf-8") as f:
        json_lib.dump(all_args, f, indent=2, ensure_ascii=False)
    log_and_save(f"✅ Arguments saved to {args_file}\n")

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        full_tokenization = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=False,
            truncation=False,
            return_length=True
        )

        for i in range(len(examples["premise"])):
            if full_tokenization['length'][i] > data_args.max_seq_length:
                print(
                    f"WARNING: Example {i} exceeds max_seq_length! Length: {full_tokenization['length'][i]}, Max: {data_args.max_seq_length}")

        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )


    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            print(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Print dataset sizes
    dataset_info = f"Dataset sizes - Train: {len(train_dataset) if training_args.do_train else 0}, " \
                   f"Val: {len(eval_dataset) if training_args.do_eval else 0}, " \
                   f"Test: {len(predict_dataset) if training_args.do_predict else 0}"
    log_and_save(dataset_info)
    
    # Add dataset file names to info
    if training_args.do_train and data_args.train_file:
        log_and_save(f"Train file: {data_args.train_file}")
    if training_args.do_eval and data_args.validation_file:
        log_and_save(f"Val file: {data_args.validation_file}")
    if training_args.do_predict and data_args.test_file:
        log_and_save(f"Test file: {data_args.test_file}")
    log_and_save("")

    # Get the metric function
    # metric = evaluate.load("xnli")
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        scores = torch.softmax(torch.tensor(predictions), axis=1) # softmax to normalize
        scores = scores[:,1] # 1 for entailment score

        # Convert predictions to probabilities for CSV
        probs_all = torch.softmax(torch.tensor(trainer.predict(predict_dataset).predictions), dim=1).numpy()
        
        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        
        # Save predictions as CSV
        import csv
        output_csv_file = os.path.join(training_args.output_dir, "predictions.csv")
        
        if trainer.is_world_process_zero():
            # Save TXT (original)
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                if data_args.get_predict_scores:
                    for index, item in enumerate(scores):
                        writer.write(f"{index}\t{item}\n")
                else:
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")
            
            # Save CSV with full details
            with open(output_csv_file, "w", newline='', encoding='utf-8') as csvfile:
                fieldnames = ['index', 'premise', 'hypothesis', 'true_label', 'predicted_label', 'prob_' + label_list[0], 'prob_' + label_list[1]]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i in range(len(predictions)):
                    premise_text = predict_dataset[i]['premise']
                    if len(premise_text) > 200:
                        premise_text = premise_text[:200] + '...'
                    
                    writer.writerow({
                        'index': i,
                        'premise': premise_text,
                        'hypothesis': predict_dataset[i]['hypothesis'],
                        'true_label': label_list[labels[i]],
                        'predicted_label': label_list[predictions[i]],
                        'prob_' + label_list[0]: float(probs_all[i][0]),
                        'prob_' + label_list[1]: float(probs_all[i][1])
                    })
            
            log_and_save(f"✅ Predictions saved to {output_predict_file}")
            log_and_save(f"✅ Predictions CSV saved to {output_csv_file}")
    
    # Save all training info to text file
    training_info_file = os.path.join(training_args.output_dir, "training_info.txt")
    with open(training_info_file, "w", encoding="utf-8") as f:
        f.write("\n".join(training_info))
    print(f"\n✅ Training info saved to {training_info_file}")


if __name__ == "__main__":
    main()

