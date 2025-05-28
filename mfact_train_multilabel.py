# #!/usr/bin/env python
# # coding=utf-8
# """
# Finetuning multi-lingual models on error-type classification.
# Adapted from Hugging Face’s run_glue.py / run_xnli.py.
# Now performs multi-label classification over summary-error tags.
# """
#
# import logging
# import os
# import random
# import sys
# from dataclasses import dataclass, field
# from typing import Optional
#
# import datasets
# import numpy as np
# from datasets import load_dataset
# import torch
# import evaluate
# import transformers
# from transformers import (
#     AutoConfig,
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     DataCollatorWithPadding,
#     EvalPrediction,
#     HfArgumentParser,
#     Trainer,
#     TrainingArguments,
#     default_data_collator,
#     set_seed,
# )
# from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
#
# # Ensure minimal versions
# check_min_version("4.25.0.dev0")
# require_version = getattr(transformers.utils.versions, "require_version", None)
# if require_version:
#     require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")
#
# logger = logging.getLogger(__name__)
#
# # -----------------------
# # Argument dataclasses
# # -----------------------
#
# @dataclass
# class DataTrainingArguments:
#     max_seq_length: Optional[int] = field(
#         default=128,
#         metadata={"help": "Max total sequence length after tokenization."},
#     )
#     overwrite_cache: bool = field(
#         default=False, metadata={"help": "Overwrite cached preprocessed datasets."}
#     )
#     pad_to_max_length: bool = field(
#         default=True,
#         metadata={"help": "Whether to pad all samples to max_seq_length."},
#     )
#     max_train_samples: Optional[int] = field(
#         default=None, metadata={"help": "Truncate number of training examples (debug)."}
#     )
#     max_eval_samples: Optional[int] = field(
#         default=None, metadata={"help": "Truncate number of eval examples (debug)."}
#     )
#     max_predict_samples: Optional[int] = field(
#         default=None, metadata={"help": "Truncate number of predict examples (debug)."}
#     )
#     train_file: Optional[str] = field(
#         default=None, metadata={"help": "Input training data file (jsonlines or csv)."}
#     )
#     validation_file: Optional[str] = field(
#         default=None, metadata={"help": "Optional evaluation data file."}
#     )
#     test_file: Optional[str] = field(
#         default=None, metadata={"help": "Optional test data file."}
#     )
#
# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(
#         default=None, metadata={"help": "Pretrained model name or path."}
#     )
#     language: Optional[str] = field(
#         default=None, metadata={"help": "Not used for multi-label; kept for compatibility."}
#     )
#     train_language: Optional[str] = field(
#         default=None, metadata={"help": "Not used here."}
#     )
#     config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path."}
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained tokenizer name or path."}
#     )
#     cache_dir: Optional[str] = field(
#         default=None, metadata={"help": "Where to store downloaded models."}
#     )
#     do_lower_case: Optional[bool] = field(
#         default=False, metadata={"help": "Lower-case tokenizer flag."}
#     )
#     use_fast_tokenizer: bool = field(
#         default=True, metadata={"help": "Use fast tokenizer."}
#     )
#     model_revision: str = field(
#         default="main", metadata={"help": "Model version to use."}
#     )
#     use_auth_token: bool = field(
#         default=False, metadata={"help": "Use HF auth token (for private models)."}
#     )
#     ignore_mismatched_sizes: bool = field(
#         default=False, metadata={"help": "Allow mismatched head sizes."}
#     )
#
# # -----------------------
# # Main
# # -----------------------
#
# def main():
#     # 1) Parse arguments
#     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#     send_example_telemetry("run_xnli_multilabel", model_args)
#
#     # 2) Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )
#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#
#     # 3) Detect last checkpoint if resuming
#     last_checkpoint = None
#     if (os.path.isdir(training_args.output_dir)
#         and training_args.do_train
#         and not training_args.overwrite_output_dir):
#         last_checkpoint = get_last_checkpoint(training_args.output_dir)
#         if last_checkpoint is None and os.listdir(training_args.output_dir):
#             raise ValueError(
#                 f"Output dir ({training_args.output_dir}) exists and is not empty."
#                 " Use --overwrite_output_dir to overwrite."
#             )
#         elif last_checkpoint:
#             logger.info(f"Resuming from checkpoint {last_checkpoint}")
#
#     set_seed(training_args.seed)
#
#     # 4) Load dataset files (CSV/JSON)
#     data_files = {}
#     extension = None
#     if data_args.train_file:
#         data_files["train"] = data_args.train_file
#         extension = data_args.train_file.split(".")[-1]
#     if data_args.validation_file:
#         data_files["validation"] = data_args.validation_file
#         extension = data_args.validation_file.split(".")[-1]
#     if data_args.test_file:
#         data_files["test"] = data_args.test_file
#         extension = data_args.test_file.split(".")[-1]
#
#     if not data_files:
#         raise ValueError("Need to specify at least one of --train_file, --validation_file, --test_file")
#
#     raw_datasets = load_dataset(
#         extension,
#         data_files=data_files,
#         cache_dir=model_args.cache_dir,
#         # use_auth_token=model_args.use_auth_token if private
#     )
#
#     # 5) Gather all error tags into label_list
#     def gather_error_types(ds):
#         s = set()
#         for ex in ds:
#             if ex.get("error_types"):
#                 for tag in ex["error_types"].split(","):
#                     s.add(tag.strip())
#         return s
#
#     all_errors = set()
#     if "train" in raw_datasets:
#         all_errors |= gather_error_types(raw_datasets["train"])
#     if "validation" in raw_datasets:
#         all_errors |= gather_error_types(raw_datasets["validation"])
#     if "test" in raw_datasets:
#         all_errors |= gather_error_types(raw_datasets["test"])
#
#     label_list = sorted(all_errors)
#     label2id = {l: i for i, l in enumerate(label_list)}
#     num_labels = len(label_list)
#     logger.info(f"Detected error types: {label_list}")
#
#     # 6) Encode error_types → multi-hot labels, drop old columns
#     def encode_error_types(example):
#         multi_hot = [0.0] * num_labels
#         if example.get("error_types"):
#             for tag in example["error_types"].split(","):
#                 tag = tag.strip()
#                 if tag in label2id:
#                     multi_hot[label2id[tag]] = 1.0
#         example["labels"] = multi_hot
#         return example
#
#     for split in ["train", "validation", "test"]:
#         if split in raw_datasets:
#             raw_datasets[split] = raw_datasets[split].map(
#                 encode_error_types,
#                 batched=False,
#                 remove_columns=["label", "error_types"],
#             )
#
#     # 7) Prepare datasets for Trainer
#     train_dataset = raw_datasets.get("train", None)
#     eval_dataset  = raw_datasets.get("validation", None)
#     predict_dataset = raw_datasets.get("test", None)
#
#     # 8) Load config/tokenizer/model
#     config = AutoConfig.from_pretrained(
#         model_args.config_name or model_args.model_name_or_path,
#         num_labels=num_labels,
#         problem_type="multi_label_classification",  # <-- multi-label flag
#         id2label={i: l for i, l in enumerate(label_list)},
#         label2id=label2id,
#         cache_dir=model_args.cache_dir,
#         revision=model_args.model_revision,
#         use_auth_token=model_args.use_auth_token,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_args.tokenizer_name or model_args.model_name_or_path,
#         do_lower_case=model_args.do_lower_case,
#         cache_dir=model_args.cache_dir,
#         use_fast=model_args.use_fast_tokenizer,
#         revision=model_args.model_revision,
#         use_auth_token=model_args.use_auth_token,
#     )
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_args.model_name_or_path,
#         config=config,
#         cache_dir=model_args.cache_dir,
#         revision=model_args.model_revision,
#         use_auth_token=model_args.use_auth_token,
#         ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
#     )
#
#     # 9) Tokenization function: warns on over-length then truncates/pads
#     padding = "max_length" if data_args.pad_to_max_length else False
#
#     def preprocess_function(examples):
#         # length check (optional warning)
#         lengths = tokenizer(
#             examples["premise"],
#             examples["hypothesis"],
#             padding=False,
#             truncation=False,
#             return_length=True,
#         )["length"]
#         for i, length in enumerate(lengths):
#             if length > data_args.max_seq_length:
#                 print(f"WARNING: Example {i} length {length} > max {data_args.max_seq_length}")
#         # now actual tokenization
#         return tokenizer(
#             examples["premise"],
#             examples["hypothesis"],
#             padding=padding,
#             max_length=data_args.max_seq_length,
#             truncation=True,
#         )
#
#     # 10) Map tokenization
#     if training_args.do_train and train_dataset is not None:
#         if data_args.max_train_samples:
#             train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
#         train_dataset = train_dataset.map(
#             preprocess_function,
#             batched=True,
#             load_from_cache_file=not data_args.overwrite_cache,
#             desc="Tokenizing train dataset",
#         )
#
#     if training_args.do_eval and eval_dataset is not None:
#         if data_args.max_eval_samples:
#             eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
#         eval_dataset = eval_dataset.map(
#             preprocess_function,
#             batched=True,
#             load_from_cache_file=not data_args.overwrite_cache,
#             desc="Tokenizing validation dataset",
#         )
#
#     if training_args.do_predict and predict_dataset is not None:
#         if data_args.max_predict_samples:
#             predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))
#         predict_dataset = predict_dataset.map(
#             preprocess_function,
#             batched=True,
#             load_from_cache_file=not data_args.overwrite_cache,
#             desc="Tokenizing test dataset",
#         )
#
#     # 11) Data collator
#     if data_args.pad_to_max_length:
#         data_collator = default_data_collator
#     elif training_args.fp16:
#         data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
#     else:
#         data_collator = None
#
#     # 12) Multi-label compute_metrics
#     from sklearn.metrics import precision_recall_fscore_support
#
#     def compute_metrics(p: EvalPrediction):
#         logits = p.predictions if not isinstance(p.predictions, tuple) else p.predictions[0]
#         probs = torch.sigmoid(torch.tensor(logits)).numpy()
#         y_true = p.label_ids
#         y_pred = (probs >= 0.5).astype(int)
#         prec, rec, f1, _ = precision_recall_fscore_support(
#             y_true, y_pred, average="micro", zero_division=0
#         )
#         return {"precision": prec, "recall": rec, "f1": f1}
#
#     # 13) Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset if training_args.do_train else None,
#         eval_dataset=eval_dataset   if training_args.do_eval  else None,
#         compute_metrics=compute_metrics,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#     )
#
#     # 14) Training / Evaluation / Prediction
#     if training_args.do_train:
#         trainer.train(resume_from_checkpoint=last_checkpoint)
#         trainer.save_model()
#     if training_args.do_eval:
#         metrics = trainer.evaluate()
#         trainer.log_metrics("eval", metrics)
#     if training_args.do_predict:
#         predictions, labels, metrics = trainer.predict(predict_dataset)
#         trainer.log_metrics("predict", metrics)
#         # write out either raw scores or thresholded labels
#         output_file = os.path.join(training_args.output_dir, "predictions.txt")
#         if trainer.is_world_process_zero():
#             with open(output_file, "w") as writer:
#                 writer.write("index\t" + "\t".join(label_list) + "\n")
#                 probs = torch.sigmoid(torch.tensor(predictions)).numpy()
#                 for i, prob in enumerate(probs):
#                     writer.write(f"{i}\t" + "\t".join(f"{x:.4f}" for x in prob) + "\n")
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# coding=utf-8
"""
Finetuning multi-lingual models on error-type classification.
Adapted from Hugging Face’s run_glue.py / run_xnli.py.
Now performs multi-label classification over summary-error tags,
with class-balanced loss and stratified oversampling.
"""

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
import torch.nn as nn
import torch.utils.data as data_utils
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

    def gather_error_types(ds):
        s = set()
        for ex in ds:
            if ex.get("error_types"):
                for tag in ex["error_types"].split(","):
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

    for split in ("train", "validation", "test"):
        if split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].map(
                encode_error_types,
                batched=False,
                remove_columns=["label", "error_types"],
            )

    train_dataset = raw_datasets.get("train", None)
    eval_dataset = raw_datasets.get("validation", None)
    predict_dataset = raw_datasets.get("test", None)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label={i: l for i, l in enumerate(label_list)},
        label2id=label2id,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    padding = "max_length" if data_args.pad_to_max_length else False

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
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing train dataset",
        ).remove_columns(["premise", "hypothesis"])

    if training_args.do_eval and eval_dataset is not None:
        if data_args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing validation dataset",
        ).remove_columns(["premise", "hypothesis"])

    if training_args.do_predict and predict_dataset is not None:
        if data_args.max_predict_samples:
            predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing test dataset",
        ).remove_columns(["premise", "hypothesis"])

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
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
    if training_args.do_predict:
        preds, labels, metrics = trainer.predict(predict_dataset)
        trainer.log_metrics("predict", metrics)
        output_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_file, "w") as writer:
                writer.write("index\t" + "\t".join(label_list) + "\n")
                probs = torch.sigmoid(torch.tensor(preds)).numpy()
                for i, row in enumerate(probs):
                    writer.write(f"{i}\t" + "\t".join(f"{x:.4f}" for x in row) + "\n")

if __name__ == "__main__":
    main()
