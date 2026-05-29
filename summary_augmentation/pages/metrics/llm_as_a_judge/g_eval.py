"""
G-EVAL: LLM-as-judge factual consistency scorer using deepeval.

Uses deepeval's GEval metric (Liu et al. 2023, arXiv 2303.16634):
  - evaluation_params: ACTUAL_OUTPUT (summary) + EXPECTED_OUTPUT (source document)
  - explicit evaluation_steps for reliable, controllable scoring
  - Rubric-guided 0–10 scoring, normalised to 0.0–1.0 via logprob-weighted sum
  - OPENAI_API_KEY must be present in environment before calling score_one/score_batch

Standalone:
    python g_eval.py --source_doc "..." --summary_text "..."
    python g_eval.py --input_csv data.csv --output_file results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ── Supported models ──────────────────────────────────────────────────────────

class SupportedModel(str, Enum):
    GPT4O_MINI  = "gpt-4o-mini"
    GPT4O       = "gpt-4o"
    GPT4_TURBO  = "gpt-4-turbo"
    GPT4        = "gpt-4"


# ── Rubric definition ─────────────────────────────────────────────────────────

@dataclass
class RubricDefinition:
    score_range: tuple[int, int]
    expected_outcome: str


FACTUAL_CONSISTENCY_STEPS: list[str] = [
    "Check whether the facts in 'actual output' (generated summary) contradict any facts in 'expected output' (source document).",
    "Heavily penalize omission of important details present in 'expected output'.",
    "Penalize any entities, numbers, dates, or names in 'actual output' that are not mentioned in 'expected output'.",
    "Vague language or contradicting opinions are OK — only penalize factual errors.",
]

FACTUAL_CONSISTENCY_RUBRIC: list[RubricDefinition] = [
    RubricDefinition(score_range=(0,  2),  expected_outcome="Factually incorrect — major hallucinations or direct contradictions."),
    RubricDefinition(score_range=(3,  6),  expected_outcome="Mostly correct but with several unsupported or inaccurate claims."),
    RubricDefinition(score_range=(7,  9),  expected_outcome="Correct but missing minor details from the source."),
    RubricDefinition(score_range=(10, 10), expected_outcome="100% faithful — every claim is grounded in the source document."),
]


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class GEvalResult:
    consistency_score: float    # 0.0–1.0 (probability-weighted by deepeval)
    evaluation_text: str        # chain-of-thought reasoning produced by deepeval
    model_name: str
    passed: bool = False        # score >= passing_threshold


@dataclass
class GEvalBatchItem:
    sample_index: int
    eval_result: GEvalResult | None = None
    error_message: str | None = None


@dataclass
class GEvalLogEntry:
    index: int
    premise: str
    hypothesis: str
    label: str
    score: float
    evaluation: str
    model: str
    passed: bool


# ── Scorer ────────────────────────────────────────────────────────────────────

class GEvalScorer:
    """
    Wraps deepeval's GEval metric for factual consistency scoring.

    A fresh GEval metric instance is created per call so that parallel
    threads never share mutable metric state (score / reason attributes).
    """

    def __init__(
        self,
        model_name: str = SupportedModel.GPT4O_MINI,
        evaluation_steps: list[str] | None = None,
        scoring_rubric: list[RubricDefinition] | None = None,
        passing_threshold: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self._evaluation_steps = evaluation_steps if evaluation_steps is not None else FACTUAL_CONSISTENCY_STEPS
        self._scoring_rubric = scoring_rubric if scoring_rubric is not None else FACTUAL_CONSISTENCY_RUBRIC
        self._passing_threshold = passing_threshold

    def _create_metric(self) -> GEval:
        return GEval(
            name="Factual Consistency",
            model=self._model_name,
            evaluation_steps=self._evaluation_steps,
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            rubric=[
                Rubric(score_range=rubric_entry.score_range, expected_outcome=rubric_entry.expected_outcome)
                for rubric_entry in self._scoring_rubric
            ],
            threshold=self._passing_threshold,
        )

    def score_one(
        self,
        source_document: str,
        generated_summary: str,
    ) -> GEvalResult:
        """Score a single pair. Raises on API or deepeval error."""
        metric = self._create_metric()
        test_case = LLMTestCase(
            input="Evaluate the factual consistency of the summary against the source document.",
            actual_output=generated_summary.strip(),
            expected_output=source_document.strip(),
        )
        metric.measure(test_case)
        return GEvalResult(
            consistency_score=metric.score,
            evaluation_text=metric.reason or "",
            model_name=self._model_name,
            passed=metric.score >= self._passing_threshold,
        )

    def _score_safely(
        self,
        pair_index: int,
        source_document: str,
        generated_summary: str,
    ) -> GEvalBatchItem:
        try:
            eval_result = self.score_one(source_document, generated_summary)
            return GEvalBatchItem(sample_index=pair_index, eval_result=eval_result)
        except Exception as scoring_error:
            logger.warning("G-EVAL failed on pair index %d: %s", pair_index, scoring_error)
            return GEvalBatchItem(sample_index=pair_index, error_message=str(scoring_error))

    def score_batch(
        self,
        document_summary_pairs: list[tuple[str, str]],
        max_parallel_workers: int = 2,
    ) -> list[GEvalBatchItem]:
        """Score pairs in parallel. Preserves input order."""
        output_items: list[GEvalBatchItem] = [
            GEvalBatchItem(sample_index=pair_index)
            for pair_index in range(len(document_summary_pairs))
        ]
        with ThreadPoolExecutor(max_workers=max_parallel_workers) as thread_pool:
            future_map = {
                thread_pool.submit(self._score_safely, pair_index, source_doc, summary_text): pair_index
                for pair_index, (source_doc, summary_text) in enumerate(document_summary_pairs)
            }
            for completed_future in as_completed(future_map):
                batch_item = completed_future.result()
                output_items[batch_item.sample_index] = batch_item
        return output_items


# ── Serialiser ────────────────────────────────────────────────────────────────

def serialize_result(
    sample_index: int,
    source_document: str,
    generated_summary: str,
    ground_truth_label: str,
    eval_result: GEvalResult,
) -> GEvalLogEntry:
    return GEvalLogEntry(
        index=sample_index,
        premise=source_document,
        hypothesis=generated_summary,
        label=ground_truth_label,
        score=eval_result.consistency_score,
        evaluation=eval_result.evaluation_text,
        model=eval_result.model_name,
        passed=eval_result.passed,
    )


# ── Standalone CLI ─────────────────────────────────────────────────────────────

def _ensure_openai_key() -> None:
    for parent_dir in Path(__file__).resolve().parents:
        env_path = parent_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break
    if not os.environ.get("OPENAI_API_KEY", ""):
        sys.exit("ERROR: OPENAI_API_KEY not found in environment or .env file.")


def _run_single_pair_mode(
    cli_args: argparse.Namespace,
    scorer: GEvalScorer,
) -> None:
    eval_result = scorer.score_one(cli_args.source_doc, cli_args.summary_text)
    print(f"Score  : {eval_result.consistency_score:.4f} / 1.0")
    print(f"Passed : {eval_result.passed}")
    print(f"Reason : {eval_result.evaluation_text}")


def _run_batch_mode(
    cli_args: argparse.Namespace,
    scorer: GEvalScorer,
) -> None:
    with open(cli_args.input_csv, encoding="utf-8") as csv_file:
        all_rows = list(csv.DictReader(csv_file))

    end_index = cli_args.end_index if cli_args.end_index != -1 else len(all_rows)
    selected_rows = all_rows[cli_args.start_index:end_index]
    document_summary_pairs = [(csv_row["premise"], csv_row["hypothesis"]) for csv_row in selected_rows]

    print(f"Scoring {len(document_summary_pairs)} samples with {cli_args.model_name} ...")
    batch_results = scorer.score_batch(document_summary_pairs, max_parallel_workers=cli_args.num_workers)

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

    valid_scores = [log_entry["score"] for log_entry in output_log if log_entry.get("score") is not None]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    print(f"Done. {len(valid_scores)}/{len(output_log)} succeeded. Mean score: {mean_score:.4f} / 1.0")
    print(f"Output: {cli_args.output_file}")


def _run_cli() -> None:
    arg_parser = argparse.ArgumentParser(description="G-EVAL factual consistency scorer (deepeval)")
    arg_parser.add_argument("--source_doc",   type=str, help="Source document text")
    arg_parser.add_argument("--summary_text", type=str, help="Generated summary text")
    arg_parser.add_argument("--input_csv",    type=str, help="CSV with 'premise' and 'hypothesis' columns")
    arg_parser.add_argument("--output_file",  type=str, default="geval_results.json")
    arg_parser.add_argument("--model_name",   type=str, default=SupportedModel.GPT4O_MINI, choices=list(SupportedModel))
    arg_parser.add_argument("--num_workers",  type=int, default=2)
    arg_parser.add_argument("--start_index",  type=int, default=0)
    arg_parser.add_argument("--end_index",    type=int, default=-1, help="-1 means all rows")
    cli_args = arg_parser.parse_args()

    _ensure_openai_key()
    scorer = GEvalScorer(model_name=cli_args.model_name)

    if cli_args.source_doc and cli_args.summary_text:
        _run_single_pair_mode(cli_args, scorer)
        return
    if cli_args.input_csv:
        _run_batch_mode(cli_args, scorer)
        return
    arg_parser.print_help()


if __name__ == "__main__":
    _run_cli()
