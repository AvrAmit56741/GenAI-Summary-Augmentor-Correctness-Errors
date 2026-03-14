"""
QAFactEval backend for Hebrew factual consistency evaluation.

Each pipeline component supports two backends — HuggingFace or OpenAI —
selectable independently at runtime.

Pipeline:
    Hebrew summary
        → (1) NP chunk extraction     (DictaBERT-syntax  | OpenAI)
        → (2) Question generation     (OpenAI            | HF multilingual)
        → (3a) IsAnsweredSumm filter  (XLM-RoBERTa-squad | OpenAI)
        → (3b) QA on source article   (XLM-RoBERTa-squad | OpenAI)

Usage:
    pipeline = PipelineFactory.hybrid(openai_client)
    result   = pipeline.run(summary, source_article)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)


# ─── Shared data classes ───────────────────────────────────────────────────────

@dataclass
class NpChunk:
    text: str


@dataclass
class GeneratedQuestion:
    question: str
    source_answer: str    # the NP chunk this question was built from


@dataclass
class QaAnswer:
    question: str
    np_chunk: str            # NP chunk that seeded the question
    summary_answer: str      # Step 3b: QA model answer from summary
    source_answer: str       # Step 4:  QA model answer from source article
    is_answerable: bool      # True if source article can answer the question
    answer_f1: float = 0.0   # token F1(summary_answer, source_answer) — the actual score


@dataclass
class QaFactEvalResult:
    questions_generated: int
    questions_after_filter: int
    qa_answers: List[QaAnswer] = field(default_factory=list)

    @property
    def f1_score(self) -> float:
        """Mean token F1 across all QA pairs (the paper's primary metric)."""
        if not self.qa_answers:
            return 0.0
        return sum(a.answer_f1 for a in self.qa_answers) / len(self.qa_answers)


# ─── Abstract base classes ─────────────────────────────────────────────────────

class AnswerSelector(ABC):
    @abstractmethod
    def extract_chunks(self, text: str) -> List[NpChunk]:
        pass


class QuestionGenerator(ABC):
    @abstractmethod
    def generate(self, summary_sentence: str, answer: NpChunk) -> Optional[GeneratedQuestion]:
        pass


class AnswerComparator(ABC):
    """Step 5: compare summary answer vs source answer → score in [0, 1]."""
    @abstractmethod
    def compare(
        self,
        question: str,
        summary_answer: str,
        source_answer: str,
        summary: str,
        source_article: str,
    ) -> float:
        pass


@dataclass
class RawQaResult:
    """Intermediate result from a single QA call (before pairing summary/source)."""
    predicted_answer: str
    is_answerable: bool


class QuestionAnswerer(ABC):
    @abstractmethod
    def answer(self, question: str, context: str) -> RawQaResult:
        pass


# ─── Answer Selection ──────────────────────────────────────────────────────────

class DictaBertAnswerSelector(AnswerSelector):
    """
    Extract NP chunks using dicta-il/dictabert-syntax dependency parser.

    Model API (from model card):
        model.predict([sentence], tokenizer)
        → [{"tree": [{"word", "dep_head_idx", "dep_func", "dep_head"}, ...]}, ...]

    For each token whose dep_func is a core nominal role, we collect its word
    plus any immediate flat/compound/amod/det/nummod dependents to form a
    multi-word NP chunk (e.g. "אפרים קישון", "מאמרים הומוריסטיים").
    """
    _MODEL_NAME = "dicta-il/dictabert-syntax"
    _NOMINAL_DEPS = {"nsubj", "obj", "iobj", "nmod", "obl", "appos", "nsubj:pass"}
    _MODIFIER_DEPS = {"flat", "compound", "amod", "det", "nummod"}

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def extract_chunks(self, text: str) -> List[NpChunk]:
        self._ensure_loaded()
        logger.info("[Step 1 / DictaBERT] Extracting NP chunks from: %s", text[:120])
        parse_result = self._model.predict([text], self._tokenizer)
        token_list = parse_result[0]["tree"]
        chunks = self._collect_np_chunks(token_list)
        logger.info("[Step 1 / DictaBERT] Extracted %d chunks: %s", len(chunks), [c.text for c in chunks])
        return chunks

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME)
        self._model = AutoModel.from_pretrained(self._MODEL_NAME, trust_remote_code=True)
        self._model.eval()

    def _collect_np_chunks(self, token_list: list) -> List[NpChunk]:
        dependents_map = self._build_dependents_map(token_list)
        chunks = []
        for head_idx, token in enumerate(token_list):
            if token.get("dep_func") not in self._NOMINAL_DEPS:
                continue
            chunk_text = self._build_chunk_text(head_idx, token_list, dependents_map)
            chunks.append(NpChunk(text=chunk_text))
        return chunks

    def _build_dependents_map(self, token_list: list) -> dict:
        n = len(token_list)
        dependents_map = {i: [] for i in range(n)}
        for idx, token in enumerate(token_list):
            head_idx = int(token.get("dep_head_idx", -1))  # cast: np.int64 → int (numpy 2.0 compat)
            dep_func = token.get("dep_func", "")
            if head_idx < 0 or head_idx >= n:
                # skip ROOT sentinel (often dep_head_idx == len(token_list))
                logger.debug("skipping out-of-range dep_head_idx=%d (n=%d) for token '%s'",
                             head_idx, n, token.get("word", "?"))
                continue
            if dep_func in self._MODIFIER_DEPS:
                dependents_map[head_idx].append(idx)
        return dependents_map

    def _build_chunk_text(self, head_idx: int, token_list: list, dependents_map: dict) -> str:
        modifier_indices = dependents_map.get(head_idx, [])
        all_indices = sorted([head_idx] + modifier_indices)
        words = [token_list[i]["word"] for i in all_indices]
        return " ".join(words)


class OpenAiAnswerSelector(AnswerSelector):
    """Extract NP chunks by prompting an OpenAI model."""

    _SYSTEM_PROMPT = (
        "You are a Hebrew linguistics expert. "
        "Extract noun phrases from Hebrew text. Return one phrase per line, nothing else."
    )

    def __init__(self, client, model_name: str = "gpt-3.5-turbo"):
        self._client = client
        self._model_name = model_name

    def extract_chunks(self, text: str) -> List[NpChunk]:
        logger.info("[Step 1 / OpenAI] Extracting NP chunks from: %s", text[:120])
        raw_response = self._call_api(text)
        chunks = self._parse_response(raw_response)
        logger.info("[Step 1 / OpenAI] Extracted %d chunks: %s", len(chunks), [c.text for c in chunks])
        return chunks

    def _call_api(self, text: str) -> str:
        user_prompt = f"Extract all noun phrases from this Hebrew text:\n\n{text}"
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def _parse_response(self, raw_response: str) -> List[NpChunk]:
        lines = [line.strip() for line in raw_response.splitlines() if line.strip()]
        return [NpChunk(text=line) for line in lines]


# ─── Question Generation ───────────────────────────────────────────────────────

class OpenAiQuestionGenerator(QuestionGenerator):
    """Generate specific Hebrew questions via OpenAI, conditioned on answer + summary."""

    _SYSTEM_PROMPT = (
        "You are a Hebrew question generation expert. "
        "Given a sentence and a target answer, generate one specific Hebrew question "
        "whose answer is exactly the target. The question must be long and extractive "
        "so it is easy to answer from a source article. Return only the question."
    )

    def __init__(self, client, model_name: str = "gpt-3.5-turbo"):
        self._client = client
        self._model_name = model_name

    def generate(self, summary_sentence: str, answer: NpChunk) -> Optional[GeneratedQuestion]:
        logger.info("[Step 2 / OpenAI] Generating question for chunk: '%s'", answer.text)
        question_text = self._call_api(summary_sentence, answer.text)
        if not question_text:
            logger.info("[Step 2 / OpenAI] No question generated for chunk: '%s'", answer.text)
            return None
        logger.info("[Step 2 / OpenAI] Generated question: %s", question_text)
        return GeneratedQuestion(question=question_text, source_answer=answer.text)

    def _call_api(self, summary_sentence: str, answer_text: str) -> str:
        user_prompt = (
            f"Sentence: {summary_sentence}\n"
            f"Target answer: {answer_text}\n"
            f"Question:"
        )
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


class HfQuestionGenerator(QuestionGenerator):
    """Generate questions via a HuggingFace text-to-text model (e.g. MixQG, mT5)."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._pipeline = None

    def generate(self, summary_sentence: str, answer: NpChunk) -> Optional[GeneratedQuestion]:
        self._ensure_loaded()
        logger.info("[Step 2 / HF] Generating question for chunk: '%s'", answer.text)
        input_text = f"answer: {answer.text}  context: {summary_sentence}"
        outputs = self._pipeline(input_text, max_length=80, num_beams=4)
        question_text = outputs[0]["generated_text"].strip()
        if not question_text:
            logger.info("[Step 2 / HF] No question generated for chunk: '%s'", answer.text)
            return None
        logger.info("[Step 2 / HF] Generated question: %s", question_text)
        return GeneratedQuestion(question=question_text, source_answer=answer.text)

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline
        self._pipeline = pipeline("text2text-generation", model=self._model_name)


# ─── Question Answering ────────────────────────────────────────────────────────

class HfQuestionAnswerer(QuestionAnswerer):
    """
    Extractive QA using deepset/xlm-roberta-large-squad2.

    Calls the model directly (bypasses transformers pipeline) to avoid a
    numpy 2.0 incompatibility in pipeline's internal postprocessing.

    Unanswerable detection (SQuAD 2.0): the model outputs high logits at
    position 0 (CLS) for unanswerable questions.  We compare the null span
    score (start[0] + end[0]) against the best non-null span score; if the
    null score wins, the question is unanswerable.
    """
    _DEFAULT_MODEL = "deepset/xlm-roberta-large-squad2"
    _NULL_SCORE_DIFF_THRESHOLD = 0.0  # null wins when null_score > best_span_score

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None

    def answer(self, question: str, context: str) -> RawQaResult:
        self._ensure_loaded()
        import torch
        inputs = self._tokenizer(
            question, context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        n = len(start_logits)

        # Best non-null span (start <= end, skip CLS at 0)
        best_score = float("-inf")
        best_start = best_end = 1
        for s in range(1, n):
            for e in range(s, min(s + 30, n)):
                score = float(start_logits[s]) + float(end_logits[e])
                if score > best_score:
                    best_score, best_start, best_end = score, s, e

        null_score = float(start_logits[0]) + float(end_logits[0])
        if null_score > best_score + self._NULL_SCORE_DIFF_THRESHOLD:
            return RawQaResult(predicted_answer="", is_answerable=False)

        ids = inputs["input_ids"][0][best_start: best_end + 1]
        answer_text = self._tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()
        return RawQaResult(predicted_answer=answer_text, is_answerable=bool(answer_text))

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForQuestionAnswering, XLMRobertaTokenizer
        self._tokenizer = XLMRobertaTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForQuestionAnswering.from_pretrained(self._model_name)
        self._model.eval()


class OpenAiQuestionAnswerer(QuestionAnswerer):
    """
    Extractive QA via OpenAI GPT with explicit unanswerable detection.
    Used for both IsAnsweredSumm filter and QA on source article.
    """
    _UNANSWERABLE_TOKEN = "UNANSWERABLE"

    _SYSTEM_PROMPT = (
        "You are a Hebrew reading comprehension expert. "
        "Answer the question using only information from the provided text. "
        f"If the text does not contain the answer, respond with exactly: UNANSWERABLE"
    )

    def __init__(self, client, model_name: str = "gpt-3.5-turbo"):
        self._client = client
        self._model_name = model_name

    def answer(self, question: str, context: str) -> RawQaResult:
        predicted = self._call_api(question, context)
        is_answerable = predicted.upper() != self._UNANSWERABLE_TOKEN
        return RawQaResult(
            predicted_answer=predicted if is_answerable else "",
            is_answerable=is_answerable,
        )

    def _call_api(self, question: str, context: str) -> str:
        user_prompt = f"Text: {context}\n\nQuestion: {question}\nAnswer:"
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# ─── Pipeline config and orchestrator ─────────────────────────────────────────

@dataclass
class PipelineConfig:
    answer_selector: AnswerSelector
    question_generator: QuestionGenerator
    question_answerer: QuestionAnswerer
    answer_comparator: AnswerComparator = None   # defaults to TokenF1Comparator if None
    isanswered_f1_threshold: float = 0.60        # Fix 1: gate F1(NP chunk, summary answer) ≥ threshold


class QaFactEvalPipeline:
    """
    Orchestrates the 4-step QAFactEval pipeline for Hebrew.

    Step 1  — Extract NP chunks from the summary (answer selection)
    Step 2  — Generate one Hebrew question per chunk
    Step 3a — IsAnsweredSumm: discard questions the summary itself cannot answer
    Step 3b — Record the summary's own answer to each surviving question
    Step 4  — Answer the same questions using the source article
    Step 5  — Score: token F1(summary_answer, source_answer)
    """

    def __init__(self, config: PipelineConfig):
        self._selector = config.answer_selector
        self._generator = config.question_generator
        self._answerer = config.question_answerer
        self._comparator = config.answer_comparator or TokenF1Comparator()
        self._isanswered_f1_threshold = config.isanswered_f1_threshold

    def warm_up(self) -> None:
        """Pre-load all lazy HF models before parallel execution to avoid race conditions."""
        for component in (self._selector, self._generator, self._answerer):
            if hasattr(component, '_ensure_loaded'):
                component._ensure_loaded()

    def run_batch(
        self,
        pairs: List[tuple],
        max_workers: int = 4,
    ) -> List[dict]:
        """
        Run the pipeline on multiple (summary, source_article) pairs in parallel.

        Pre-warms all HF models first so worker threads never call _ensure_loaded
        concurrently.  Each dict in the returned list has keys:
            index, result (QaFactEvalResult | None), error (str | None)
        Results are returned in input order.
        """
        self.warm_up()
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_one(idx: int, summary: str, source: str) -> dict:
            try:
                return {"index": idx, "result": self.run(summary, source), "error": None}
            except Exception as exc:
                import traceback
                full_tb = traceback.format_exc()
                logger.error("QAFactEval error on sample %d:\n%s", idx, full_tb)
                return {"index": idx, "result": None, "error": full_tb}

        slots: List[dict] = [{}] * len(pairs)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_one, i, summary, source): i
                for i, (summary, source) in enumerate(pairs)
            }
            for future in as_completed(futures):
                item = future.result()
                slots[item["index"]] = item
        return slots

    def run(self, summary: str, source_article: str) -> QaFactEvalResult:
        logger.info("=== QAFactEval START ===")
        logger.info("Summary: %s", summary[:150])
        chunks = self._selector.extract_chunks(summary)
        questions = self._generate_questions(summary, chunks)
        logger.info("[Step 2] Generated %d questions total", len(questions))
        filtered, summary_answers = self._filter_and_answer_summary(questions, summary)
        logger.info("[Step 3a] %d questions survived IsAnsweredSumm filter (dropped %d)",
                    len(filtered), len(questions) - len(filtered))
        qa_answers = self._answer_from_source(filtered, summary_answers, summary, source_article)
        result = QaFactEvalResult(
            questions_generated=len(questions),
            questions_after_filter=len(filtered),
            qa_answers=qa_answers,
        )
        logger.info("=== QAFactEval END — f1_score=%.4f ===", result.f1_score)
        return result

    def _generate_questions(self, summary: str, chunks: List[NpChunk]) -> List[GeneratedQuestion]:
        generated = []
        for chunk in chunks:
            question = self._generator.generate(summary, chunk)
            if question:
                generated.append(question)
        return generated

    def _filter_and_answer_summary(
        self, questions: List[GeneratedQuestion], summary: str
    ) -> tuple:
        """
        Step 3a+3b: filter questions the summary cannot answer,
        and return the summary's answer for each surviving question.
        Returns (filtered_questions, summary_answers).
        """
        filtered, summary_answers = [], []
        for question in questions:
            logger.info("[Step 3a] Checking answerability from SUMMARY: '%s'", question.question)
            qa = self._answerer.answer(question.question, summary)
            if not qa.is_answerable:
                logger.info("[Step 3a] ✗ Gate 1 failed — not answerable from summary — discarded")
                continue
            # Fix 1 — Gate 2: F1(NP chunk, summary answer) must meet threshold
            f1 = _token_f1(question.source_answer, qa.predicted_answer)
            if f1 < self._isanswered_f1_threshold:
                logger.info(
                    "[Step 3a] ✗ Gate 2 failed — F1(NP chunk, summary answer)=%.2f < threshold=%.2f — discarded",
                    f1, self._isanswered_f1_threshold,
                )
                continue
            logger.info(
                "[Step 3a] ✓ Both gates passed — summary answer: '%s' | F1(chunk, answer)=%.2f",
                qa.predicted_answer, f1,
            )
            filtered.append(question)
            summary_answers.append(qa.predicted_answer)
        return filtered, summary_answers

    def _answer_from_source(
        self,
        questions: List[GeneratedQuestion],
        summary_answers: List[str],
        summary: str,
        source_article: str,
    ) -> List[QaAnswer]:
        """Step 4+5: answer from source, then compare answers via configured comparator."""
        answers = []
        for question, summary_ans in zip(questions, summary_answers):
            logger.info("[Step 4] Answering from SOURCE: '%s'", question.question)
            qa = self._answerer.answer(question.question, source_article)
            # Fix 2 — Answerability Penalty: unanswerable source forces score = 0
            if not qa.is_answerable:
                score = 0.0
                logger.info(
                    "[Step 5] Source unanswerable — Answerability Penalty applied (score=0.0) | NP chunk: '%s'",
                    question.source_answer,
                )
            else:
                # Paper §4 (line 81): overlap is between the *selected answer* (NP chunk)
                # and the QA model output from the source — NOT between QA(summary) and QA(source).
                score = self._comparator.compare(
                    question.question, question.source_answer, qa.predicted_answer, summary, source_article
                )
                logger.info("[Step 5] NP chunk: '%s' | Source answer: '%s' | Score: %.4f",
                            question.source_answer, qa.predicted_answer, score)
            answers.append(QaAnswer(
                question=question.question,
                np_chunk=question.source_answer,
                summary_answer=summary_ans,
                source_answer=qa.predicted_answer,
                is_answerable=qa.is_answerable,
                answer_f1=score,
            ))
        return answers


# ─── Answer Comparators ────────────────────────────────────────────────────────

class TokenF1Comparator(AnswerComparator):
    """Token-level F1 between summary answer and source answer."""

    def compare(self, question, summary_answer, source_answer, summary, source_article) -> float:
        return _token_f1(summary_answer, source_answer)


class ExactMatchComparator(AnswerComparator):
    """Returns 1.0 if normalized answers match exactly, 0.0 otherwise."""

    @staticmethod
    def _normalize(text: str) -> str:
        import re
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        return " ".join(text.split())

    def compare(self, question, summary_answer, source_answer, summary, source_article) -> float:
        score = 1.0 if self._normalize(summary_answer) == self._normalize(source_answer) else 0.0
        logger.info("[Step 5 / EM] '%s' vs '%s' → %.1f", summary_answer, source_answer, score)
        return score


class OpenAiAnswerComparator(AnswerComparator):
    """
    GPT-based replacement for the LERC learned answer-overlap model.

    LERC (Chen et al. 2020) takes: (question, context/passage, reference_answer, candidate_answer)
    and scores 1-5 how well the candidate matches the reference.

    Here we replicate that task with a YES/NO prompt:
    - reference_answer = NP chunk selected from the summary (the "correct" answer)
    - candidate_answer = QA model output extracted from the source article
    - context         = source article (the passage the QA model read)

    The summary is intentionally NOT included — LERC does not see it.
    Returns 1.0 (equivalent) or 0.0 (not equivalent).
    """

    _SYSTEM_PROMPT = (
        "You are an answer equivalence judge for Hebrew reading comprehension. "
        "You will be given a question, a passage (source article), a reference answer, "
        "and a candidate answer extracted from the passage. "
        "Decide whether the candidate answer is semantically equivalent to the reference answer "
        "in the context of the question. "
        "Reply with exactly one word: YES or NO."
    )

    def __init__(self, client, model_name: str = "gpt-4o-mini"):
        self._client = client
        self._model_name = model_name

    def compare(self, question, summary_answer, source_answer, summary, source_article) -> float:
        # summary_answer here is the NP chunk (selected answer) — see _answer_from_source.
        # source_answer is the QA model output from the source article.
        verdict = self._call_api(question, summary_answer, source_answer, source_article)
        score = 1.0 if verdict.strip().upper().startswith("YES") else 0.0
        logger.info("[Step 5 / GPT Judge] ref='%s' | cand='%s' → %s (%.1f)",
                    summary_answer, source_answer, verdict.strip(), score)
        return score

    def _call_api(self, question, reference_answer, candidate_answer, source_article) -> str:
        user_prompt = (
            f"Passage (source article):\n{source_article[:1500]}\n\n"
            f"Question: {question}\n\n"
            f"Reference answer (from summary): {reference_answer}\n"
            f"Candidate answer (extracted from passage): {candidate_answer}\n\n"
            f"Is the candidate answer semantically equivalent to the reference answer? Reply YES or NO."
        )
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


def _token_f1(reference: str, prediction: str) -> float:
    """Token-level F1 between reference (NP chunk) and predicted answer."""
    if not reference or not prediction:
        return 0.0
    ref_tokens = set(reference.strip().split())
    pred_tokens = set(prediction.strip().split())
    common = ref_tokens & pred_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ─── Factory ───────────────────────────────────────────────────────────────────

class PipelineFactory:
    """Convenience factory for common pipeline configurations."""

    @staticmethod
    def hybrid(openai_client, openai_model: str = "gpt-3.5-turbo") -> QaFactEvalPipeline:
        """Recommended: DictaBERT parsing + OpenAI QG + HF QA."""
        config = PipelineConfig(
            answer_selector=DictaBertAnswerSelector(),
            question_generator=OpenAiQuestionGenerator(openai_client, openai_model),
            question_answerer=HfQuestionAnswerer(),
        )
        return QaFactEvalPipeline(config)

    @staticmethod
    def full_openai(openai_client, openai_model: str = "gpt-3.5-turbo") -> QaFactEvalPipeline:
        """All steps via OpenAI — no local models required."""
        config = PipelineConfig(
            answer_selector=OpenAiAnswerSelector(openai_client, openai_model),
            question_generator=OpenAiQuestionGenerator(openai_client, openai_model),
            question_answerer=OpenAiQuestionAnswerer(openai_client, openai_model),
        )
        return QaFactEvalPipeline(config)

    @staticmethod
    def full_hf(qg_model_name: str = "nielsr/MixQG-large") -> QaFactEvalPipeline:
        """All steps via HuggingFace — no API key required."""
        config = PipelineConfig(
            answer_selector=DictaBertAnswerSelector(),
            question_generator=HfQuestionGenerator(qg_model_name),
            question_answerer=HfQuestionAnswerer(),
        )
        return QaFactEvalPipeline(config)
