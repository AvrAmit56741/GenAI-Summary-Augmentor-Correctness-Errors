"""
BARTScore: factual consistency scorer using BART generation log-probability.

Computes P(summary | source) by force-decoding the summary through BART and
averaging the log-probabilities of each summary token.

  raw_score       : average log-prob per token  (−∞ to 0,   closer to 0 = better)
  normalized_score: exp(raw_score)              (0.0 to 1.0, higher     = better)

Reference: Yuan et al. 2021 — "BARTScore: Evaluating Generated Text as Text Generation"
           NeurIPS 2021.  https://arxiv.org/abs/2106.11520

Standalone:
    python bartscore.py --source_doc "..." --summary_text "..."
    python bartscore.py --input_csv data.csv --output_file results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)


# ── Supported models ──────────────────────────────────────────────────────────

class SupportedModel(str, Enum):
    BART_LARGE_CNN  = "facebook/bart-large-cnn"   # fine-tuned on CNN/DM — best for news
    BART_LARGE      = "facebook/bart-large"        # base pretrained BART
    BART_LARGE_MNLI = "facebook/bart-large-mnli"   # NLI fine-tuned variant


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class BartScoreResult:
    raw_score: float        # average log-prob per token (−∞ to 0, closer to 0 = better)
    normalized_score: float # exp(raw_score) → 0.0–1.0
    model_name: str
    passed: bool = False    # normalized_score >= passing_threshold


@dataclass
class BartScoreBatchItem:
    sample_index: int
    eval_result: BartScoreResult | None = None
    error_message: str | None = None


@dataclass
class BartScoreLogEntry:
    index: int
    premise: str
    hypothesis: str
    label: str
    raw_score: float
    normalized_score: float
    model: str
    passed: bool


# ── Scorer ────────────────────────────────────────────────────────────────────

class BartScorer:
    """
    BARTScore factual consistency scorer.

    Construct via BartScorer.from_pretrained() to load weights, or inject
    an already-loaded model and tokenizer directly for testing.
    Scores pairs in GPU mini-batches — not thread-parallelised.
    """

    def __init__(
        self,
        bart_model: BartForConditionalGeneration,
        bart_tokenizer: BartTokenizer,
        compute_device: str,
        model_name: str = SupportedModel.BART_LARGE_CNN,
        batch_size: int = 4,
        max_source_length: int = 1024,
        max_summary_length: int = 128,
        passing_threshold: float = 0.1,
    ) -> None:
        self._model = bart_model
        self._tokenizer = bart_tokenizer
        self._device = compute_device
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_source_length = max_source_length
        self._max_summary_length = max_summary_length
        self._passing_threshold = passing_threshold

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = SupportedModel.BART_LARGE_CNN,
        compute_device: str | None = None,
        batch_size: int = 4,
        max_source_length: int = 1024,
        max_summary_length: int = 128,
        passing_threshold: float = 0.1,
    ) -> BartScorer:
        """Load BART weights from HuggingFace and construct a ready scorer."""
        resolved_device = compute_device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading BARTScore model %s on %s", model_name, resolved_device)
        bart_tokenizer = BartTokenizer.from_pretrained(model_name)
        bart_model = BartForConditionalGeneration.from_pretrained(model_name)
        bart_model.eval()
        bart_model.to(resolved_device)
        return cls(
            bart_model=bart_model,
            bart_tokenizer=bart_tokenizer,
            compute_device=resolved_device,
            model_name=model_name,
            batch_size=batch_size,
            max_source_length=max_source_length,
            max_summary_length=max_summary_length,
            passing_threshold=passing_threshold,
        )

    def _compute_raw_scores(
        self,
        source_documents: list[str],
        generated_summaries: list[str],
    ) -> list[float]:
        """Force-decode summaries given sources; return average log-prob per token."""
        source_encodings = self._tokenizer(
            source_documents, return_tensors="pt", truncation=True,
            max_length=self._max_source_length, padding=True,
        )
        target_encodings = self._tokenizer(
            generated_summaries, return_tensors="pt", truncation=True,
            max_length=self._max_summary_length, padding=True,
        )
        input_ids = source_encodings["input_ids"].to(self._device)
        attention_mask = source_encodings["attention_mask"].to(self._device)

        target_labels = target_encodings["input_ids"].clone().to(self._device)
        target_labels[target_labels == self._tokenizer.pad_token_id] = -100

        with torch.no_grad():
            model_outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_labels,
            )

        batch_size, _, vocab_size = model_outputs.logits.shape
        loss_fn = CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fn(
            model_outputs.logits.view(-1, vocab_size),
            target_labels.view(-1),
        ).view(batch_size, -1)

        token_counts = (target_labels != -100).sum(dim=1).float().clamp(min=1)
        return (-per_token_loss.sum(dim=1) / token_counts).cpu().tolist()

    def _build_result(
        self,
        raw_score: float,
    ) -> BartScoreResult:
        normalized_score = math.exp(raw_score)
        return BartScoreResult(
            raw_score=raw_score,
            normalized_score=normalized_score,
            model_name=self._model_name,
            passed=normalized_score >= self._passing_threshold,
        )

    def _process_batch_chunk(
        self,
        batch_start: int,
        batch_sources: list[str],
        batch_summaries: list[str],
    ) -> list[BartScoreBatchItem]:
        try:
            raw_scores = self._compute_raw_scores(batch_sources, batch_summaries)
            return [
                BartScoreBatchItem(
                    sample_index=batch_start + chunk_index,
                    eval_result=self._build_result(raw_score),
                )
                for chunk_index, raw_score in enumerate(raw_scores)
            ]
        except (RuntimeError, ValueError) as batch_error:
            logger.warning("BARTScore failed on batch at index %d: %s", batch_start, batch_error)
            return [
                BartScoreBatchItem(
                    sample_index=batch_start + chunk_index,
                    error_message=str(batch_error),
                )
                for chunk_index in range(len(batch_sources))
            ]

    def score_one(
        self,
        source_document: str,
        generated_summary: str,
    ) -> BartScoreResult:
        """Score a single pair. Raises on model or tokenization error."""
        raw_scores = self._compute_raw_scores(
            source_documents=[source_document.strip()],
            generated_summaries=[generated_summary.strip()],
        )
        return self._build_result(raw_scores[0])

    def score_batch(
        self,
        document_summary_pairs: list[tuple[str, str]],
        max_parallel_workers: int = 1,
    ) -> list[BartScoreBatchItem]:
        """Score all pairs in GPU mini-batches. Preserves input order."""
        source_documents = [pair[0].strip() for pair in document_summary_pairs]
        generated_summaries = [pair[1].strip() for pair in document_summary_pairs]
        output_items: list[BartScoreBatchItem] = []
        for batch_start in range(0, len(document_summary_pairs), self._batch_size):
            batch_sources = source_documents[batch_start:batch_start + self._batch_size]
            batch_summaries = generated_summaries[batch_start:batch_start + self._batch_size]
            output_items.extend(self._process_batch_chunk(batch_start, batch_sources, batch_summaries))
        return output_items


# ── Serialiser ────────────────────────────────────────────────────────────────

def serialize_result(
    sample_index: int,
    source_document: str,
    generated_summary: str,
    ground_truth_label: str,
    eval_result: BartScoreResult,
) -> BartScoreLogEntry:
    return BartScoreLogEntry(
        index=sample_index,
        premise=source_document,
        hypothesis=generated_summary,
        label=ground_truth_label,
        raw_score=eval_result.raw_score,
        normalized_score=eval_result.normalized_score,
        model=eval_result.model_name,
        passed=eval_result.passed,
    )


# ── Standalone CLI ─────────────────────────────────────────────────────────────

def _ensure_environment() -> None:
    for parent_dir in Path(__file__).resolve().parents:
        env_path = parent_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break


def _run_single_pair_mode(
    cli_args: argparse.Namespace,
    scorer: BartScorer,
) -> None:
    eval_result = scorer.score_one(cli_args.source_doc, cli_args.summary_text)
    print(f"Raw score  : {eval_result.raw_score:.4f}  (avg log-prob per token)")
    print(f"Normalized : {eval_result.normalized_score:.4f} / 1.0  (exp of raw score)")
    print(f"Passed     : {eval_result.passed}")


def _run_batch_mode(
    cli_args: argparse.Namespace,
    scorer: BartScorer,
) -> None:
    with open(cli_args.input_csv, encoding="utf-8") as csv_file:
        all_rows = list(csv.DictReader(csv_file))

    end_index = cli_args.end_index if cli_args.end_index != -1 else len(all_rows)
    selected_rows = all_rows[cli_args.start_index:end_index]
    document_summary_pairs = [(csv_row["premise"], csv_row["hypothesis"]) for csv_row in selected_rows]

    print(f"Scoring {len(document_summary_pairs)} samples with {cli_args.model_name} ...")
    batch_results = scorer.score_batch(document_summary_pairs)

    output_log = []
    for row_index, batch_item in enumerate(batch_results):
        csv_row = selected_rows[row_index]
        if batch_item.error_message:
            output_log.append({"index": cli_args.start_index + row_index, "error": batch_item.error_message})
        else:
            output_log.append(asdict(serialize_result(
                sample_index=cli_args.start_index + row_index,
                source_document=csv_row["premise"],
                generated_summary=csv_row["hypothesis"],
                ground_truth_label=csv_row.get("label", ""),
                eval_result=batch_item.eval_result,
            )))

    with open(cli_args.output_file, "w", encoding="utf-8") as out_file:
        json.dump(output_log, out_file, ensure_ascii=False, indent=2)

    valid_scores = [log_entry["normalized_score"] for log_entry in output_log if log_entry.get("normalized_score") is not None]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    print(f"Done. {len(valid_scores)}/{len(output_log)} succeeded. Mean normalized score: {mean_score:.4f}")
    print(f"Output: {cli_args.output_file}")


def _run_cli() -> None:
    arg_parser = argparse.ArgumentParser(description="BARTScore factual consistency scorer")
    arg_parser.add_argument("--source_doc",   type=str, help="Source document text")
    arg_parser.add_argument("--summary_text", type=str, help="Generated summary text")
    arg_parser.add_argument("--input_csv",    type=str, help="CSV with 'premise' and 'hypothesis' columns")
    arg_parser.add_argument("--output_file",  type=str, default="bartscore_results.json")
    arg_parser.add_argument("--model_name",   type=str, default=SupportedModel.BART_LARGE_CNN, choices=list(SupportedModel))
    arg_parser.add_argument("--batch_size",   type=int, default=4)
    arg_parser.add_argument("--start_index",  type=int, default=0)
    arg_parser.add_argument("--end_index",    type=int, default=-1, help="-1 means all rows")
    cli_args = arg_parser.parse_args()

    _ensure_environment()
    scorer = BartScorer.from_pretrained(
        model_name=cli_args.model_name,
        batch_size=cli_args.batch_size,
    )

    if cli_args.source_doc and cli_args.summary_text:
        _run_single_pair_mode(cli_args, scorer)
        return
    if cli_args.input_csv:
        _run_batch_mode(cli_args, scorer)
        return
    arg_parser.print_help()


if __name__ == "__main__":
    _run_cli()
