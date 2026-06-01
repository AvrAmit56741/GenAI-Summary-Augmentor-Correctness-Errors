"""
Vanilla Zero-Shot LLM-as-Judge factual consistency scorer.

Direct OpenAI chat completion — scores 1–5 by parsing the model's text reply.
No logprobs, no probability weighting. Simpler than G-EVAL but less precise.

Compare with g_eval.py: this file parses a single digit from the reply text;
g_eval.py reads the probability distribution over all score tokens (logprobs).

OPENAI_API_KEY must be present in the environment before constructing VanillaScorer.

Standalone:
    python vanila_zero_shot.py --source_doc "..." --summary_text "..."
    python vanila_zero_shot.py --input_csv data.csv --output_file results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError

_MAX_SCORE_RETRIES = 3
_RETRY_BASE_DELAY_SECONDS = 2.0

logger = logging.getLogger(__name__)


# ── Supported models ──────────────────────────────────────────────────────────

class SupportedModel(str, Enum):
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O      = "gpt-4o"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4       = "gpt-4"


# ── Prompt template ───────────────────────────────────────────────────────────

VANILLA_SYSTEM_PROMPT: str = (
    "You are an expert evaluator for text summarization factual consistency.\n\n"
    "Given a source document and a generated summary, evaluate whether every "
    "claim in the summary is supported by the source document.\n\n"
    "Scoring scale:\n"
    "  1 — Major hallucinations or direct contradictions of the source\n"
    "  2 — Several unsupported claims; partially inconsistent with source\n"
    "  3 — Mostly consistent; minor unsupported details present\n"
    "  4 — Consistent with at most one minor unsupported detail\n"
    "  5 — Fully faithful; every claim is grounded in the source\n\n"
    "The text may be in Hebrew or English — evaluate in the language of the source.\n\n"
    "Respond in exactly this format:\n"
    "Reasoning: <your 2–3 sentence explanation>\n"
    "Score: <single digit 1–5>"
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class VanillaResult:
    raw_score: int           # 1–5 integer parsed from model reply
    score_normalized: float  # (raw_score - 1) / 4  →  0.0–1.0
    reasoning_text: str
    model_name: str
    passed: bool = False     # score_normalized >= passing_threshold


@dataclass
class VanillaBatchItem:
    sample_index: int
    eval_result: VanillaResult | None = None
    error_message: str | None = None


@dataclass
class VanillaLogEntry:
    index: int
    premise: str
    hypothesis: str
    label: str
    score: int
    score_normalized: float
    reasoning: str
    model: str
    passed: bool


# ── Scorer ────────────────────────────────────────────────────────────────────

class VanillaScorer:
    """
    Direct OpenAI LLM-as-judge scorer — no logprobs, parses a 1–5 integer from text.
    openai_client is injected so the caller controls API key and retry config.
    A fresh API call is made per score_one(); thread-safe by design.
    """

    def __init__(
        self,
        openai_client: OpenAI,
        model_name: str = SupportedModel.GPT4O_MINI,
        system_prompt: str = VANILLA_SYSTEM_PROMPT,
        passing_threshold: float = 0.5,
    ) -> None:
        self._openai_client = openai_client
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._passing_threshold = passing_threshold

    def _build_user_message(
        self,
        source_document: str,
        generated_summary: str,
    ) -> str:
        return (
            f"Source document:\n{source_document}\n\n"
            f"Generated summary:\n{generated_summary}"
        )

    def _parse_score(
        self,
        response_text: str,
    ) -> int:
        score_match = re.search(r"Score:\s*([1-5])", response_text)
        if not score_match:
            raise ValueError(f"No valid 1–5 score found in model response: {response_text!r}")
        return int(score_match.group(1))

    def _parse_reasoning(
        self,
        response_text: str,
    ) -> str:
        reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=\nScore:)", response_text, re.DOTALL)
        return reasoning_match.group(1).strip() if reasoning_match else response_text.strip()

    def _score_once(
        self,
        source_document: str,
        generated_summary: str,
    ) -> VanillaResult:
        """Single API call + parse attempt. Raises OpenAIError or ValueError on failure."""
        user_message = self._build_user_message(
            source_document=source_document.strip(),
            generated_summary=generated_summary.strip(),
        )
        api_response = self._openai_client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
        )
        response_text = api_response.choices[0].message.content or ""
        raw_score = self._parse_score(response_text)
        score_normalized = (raw_score - 1) / 4
        return VanillaResult(
            raw_score=raw_score,
            score_normalized=score_normalized,
            reasoning_text=self._parse_reasoning(response_text),
            model_name=self._model_name,
            passed=score_normalized >= self._passing_threshold,
        )

    def score_one(
        self,
        source_document: str,
        generated_summary: str,
    ) -> VanillaResult:
        """Score a single pair with automatic retry.

        Retries up to _MAX_SCORE_RETRIES times:
          - RateLimitError / OpenAIError → exponential backoff before retry
          - ValueError (parse failure)   → immediate retry, no sleep
        Raises the last exception if all attempts fail.
        """
        last_error: Exception | None = None
        for attempt_number in range(_MAX_SCORE_RETRIES):
            try:
                return self._score_once(source_document, generated_summary)
            except RateLimitError as rate_error:
                last_error = rate_error
                delay = _RETRY_BASE_DELAY_SECONDS * (2 ** attempt_number)
                logger.warning(
                    "Rate limit on attempt %d/%d — retrying in %.1fs: %s",
                    attempt_number + 1, _MAX_SCORE_RETRIES, delay, rate_error,
                )
                time.sleep(delay)
            except OpenAIError as api_error:
                last_error = api_error
                delay = _RETRY_BASE_DELAY_SECONDS * (2 ** attempt_number)
                logger.warning(
                    "API error on attempt %d/%d — retrying in %.1fs: %s",
                    attempt_number + 1, _MAX_SCORE_RETRIES, delay, api_error,
                )
                time.sleep(delay)
            except ValueError as parse_error:
                last_error = parse_error
                logger.warning(
                    "Parse failure on attempt %d/%d — retrying immediately: %s",
                    attempt_number + 1, _MAX_SCORE_RETRIES, parse_error,
                )
        raise last_error

    def _score_safely(
        self,
        pair_index: int,
        source_document: str,
        generated_summary: str,
    ) -> VanillaBatchItem:
        try:
            eval_result = self.score_one(source_document, generated_summary)
            return VanillaBatchItem(sample_index=pair_index, eval_result=eval_result)
        except (OpenAIError, ValueError) as scoring_error:
            logger.warning("Vanilla scorer failed on pair index %d: %s", pair_index, scoring_error)
            return VanillaBatchItem(sample_index=pair_index, error_message=str(scoring_error))

    def score_batch(
        self,
        document_summary_pairs: list[tuple[str, str]],
        max_parallel_workers: int = 2,
    ) -> list[VanillaBatchItem]:
        """Score pairs in parallel. Preserves input order."""
        output_items: list[VanillaBatchItem] = [
            VanillaBatchItem(sample_index=pair_index)
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
    eval_result: VanillaResult,
) -> VanillaLogEntry:
    return VanillaLogEntry(
        index=sample_index,
        premise=source_document,
        hypothesis=generated_summary,
        label=ground_truth_label,
        score=eval_result.raw_score,
        score_normalized=eval_result.score_normalized,
        reasoning=eval_result.reasoning_text,
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
    scorer: VanillaScorer,
) -> None:
    eval_result = scorer.score_one(cli_args.source_doc, cli_args.summary_text)
    print(f"Score      : {eval_result.raw_score} / 5")
    print(f"Normalized : {eval_result.score_normalized:.4f} / 1.0")
    print(f"Passed     : {eval_result.passed}")
    print(f"Reasoning  : {eval_result.reasoning_text}")


def _run_batch_mode(
    cli_args: argparse.Namespace,
    scorer: VanillaScorer,
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

    valid_scores = [
        log_entry["score_normalized"]
        for log_entry in output_log
        if log_entry.get("score_normalized") is not None
    ]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    print(f"Done. {len(valid_scores)}/{len(output_log)} succeeded. Mean score: {mean_score:.4f} / 1.0")
    print(f"Output: {cli_args.output_file}")


def _run_cli() -> None:
    arg_parser = argparse.ArgumentParser(description="Vanilla LLM-as-judge factual consistency scorer")
    arg_parser.add_argument("--source_doc",   type=str, help="Source document text")
    arg_parser.add_argument("--summary_text", type=str, help="Generated summary text")
    arg_parser.add_argument("--input_csv",    type=str, help="CSV with 'premise' and 'hypothesis' columns")
    arg_parser.add_argument("--output_file",  type=str, default="vanilla_results.json")
    arg_parser.add_argument("--model_name",   type=str, default=SupportedModel.GPT4O_MINI, choices=list(SupportedModel))
    arg_parser.add_argument("--num_workers",  type=int, default=2)
    arg_parser.add_argument("--start_index",  type=int, default=0)
    arg_parser.add_argument("--end_index",    type=int, default=-1, help="-1 means all rows")
    cli_args = arg_parser.parse_args()

    _ensure_openai_key()
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    scorer = VanillaScorer(openai_client=openai_client, model_name=cli_args.model_name)

    if cli_args.source_doc and cli_args.summary_text:
        _run_single_pair_mode(cli_args, scorer)
        return
    if cli_args.input_csv:
        _run_batch_mode(cli_args, scorer)
        return
    arg_parser.print_help()


if __name__ == "__main__":
    _run_cli()
