"""
SummaC backend — NLI-based factual consistency for summarization.

Implements two model variants from Laban et al. (2021):
  - SummacZS  : zero-shot, col-max → mean, no training required
  - SummacConv: trained 1-D convolution over score histograms (H=50 bins)

Pipeline:
    (source_doc, summary)
        → sentence splitting
        → NLI pair matrix   (M doc-sentences × N summary-sentences)
        → aggregation       (ZS: col-max→mean | Conv: histogram→conv→mean)
        → final score       (scalar ∈ [0,1]) → threshold → binary label
"""
from __future__ import annotations

import logging
import re
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SentenceScore:
    """Score breakdown for a single summary sentence."""
    sentence_text: str
    raw_scores: List[float]   # entailment score from each doc sentence (column of NLI matrix)
    final_score: float        # after aggregation: max (ZS) or conv output (Conv)


@dataclass
class SummacResult:
    """Full pipeline result for one (source_doc, summary) pair."""
    doc_sentences: List[str]
    summary_sentences: List[str]
    nli_matrix: List[List[float]]           # shape [M_doc][N_summ], row=doc, col=summary
    sentence_scores: List[SentenceScore]    # one per summary sentence
    summary_score: float                    # mean of sentence final_scores
    is_consistent: bool
    threshold_used: float


# ── Sentence splitting ────────────────────────────────────────────────────────

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')


def split_sentences(input_text: str) -> List[str]:
    """Split text into non-empty sentences on punctuation boundaries."""
    parts = _SENTENCE_BOUNDARY.split(input_text.strip())
    return [s.strip() for s in parts if s.strip()]


# ── NLI Scorer ────────────────────────────────────────────────────────────────

class NliScorer(ABC):
    """Returns entailment probability ∈ [0,1] for each (premise, hypothesis) pair."""

    @abstractmethod
    def score_pairs(
        self,
        premise_texts: List[str],
        hypothesis_texts: List[str],
    ) -> List[float]:
        pass

    def warm_up(self) -> None:
        """Pre-load model weights before parallel inference."""


class HfNliScorer(NliScorer):
    """
    HuggingFace text-classification NLI model.

    Automatically detects the entailment label from model output.
    Supports batched inference to reduce API overhead.

    Recommended models:
      - roberta-large-mnli                     (English, strong)
      - cross-encoder/nli-deberta-v3-base      (English, strongest)
      - MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingual, Hebrew-capable)
      - facebook/bart-large-mnli               (English, zero-shot capable)
    """

    SUPPORTED_MODELS = [
        "roberta-large-mnli",
        "cross-encoder/nli-deberta-v3-base",
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "facebook/bart-large-mnli",
        "typeform/distilbert-base-uncased-mnli",
    ]

    def __init__(self, model_name: str, batch_size: int = 16):
        self._model_name = model_name
        self._batch_size = batch_size
        self._pipeline = None

    def score_pairs(
        self,
        premise_texts: List[str],
        hypothesis_texts: List[str],
    ) -> List[float]:
        self._ensure_loaded()
        all_scores: List[float] = []
        pairs = list(zip(premise_texts, hypothesis_texts))
        for start_idx in range(0, len(pairs), self._batch_size):
            batch_pairs = pairs[start_idx: start_idx + self._batch_size]
            batch_scores = self._score_batch(batch_pairs)
            all_scores.extend(batch_scores)
        return all_scores

    def _score_batch(self, pair_batch: List[Tuple[str, str]]) -> List[float]:
        inputs = [{"text": premise, "text_pair": hypo} for premise, hypo in pair_batch]
        raw_outputs = self._pipeline(inputs)
        return [self._extract_entailment(output) for output in raw_outputs]

    def _extract_entailment(self, model_output: list) -> float:
        """Find the entailment class score regardless of label naming convention."""
        for label_item in model_output:
            if "entail" in label_item["label"].lower():
                return float(label_item["score"])
        logger.warning("[NLI] No entailment label found in output: %s", model_output)
        return max(item["score"] for item in model_output)

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline
        logger.info("[NLI] Loading model: %s", self._model_name)
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self._model_name,
            top_k=None,          # return all label scores
            truncation=True,
            max_length=512,
        )
        logger.info("[NLI] Model ready: %s", self._model_name)

    def warm_up(self) -> None:
        self._ensure_loaded()


# ── NLI Pair Matrix ───────────────────────────────────────────────────────────

class PairMatrixBuilder:
    """Builds an M×N entailment matrix for (doc_sentences, summary_sentences)."""

    def __init__(self, nli_scorer: NliScorer):
        self._scorer = nli_scorer

    def build(
        self,
        doc_sentences: List[str],
        summary_sentences: List[str],
    ) -> np.ndarray:
        """Return float32 array of shape (M_doc, N_summary)."""
        logger.info("[Matrix] building %d×%d NLI matrix", len(doc_sentences), len(summary_sentences))
        premises, hypotheses = self._flatten_pairs(doc_sentences, summary_sentences)
        flat_scores = self._scorer.score_pairs(premises, hypotheses)
        matrix = np.array(flat_scores, dtype=np.float32).reshape(
            len(doc_sentences), len(summary_sentences)
        )
        logger.debug("[Matrix]\n%s", matrix)
        return matrix

    def _flatten_pairs(
        self,
        doc_sentences: List[str],
        summary_sentences: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Enumerate all (Dᵢ, Sⱼ) combinations row-major."""
        premises, hypotheses = [], []
        for doc_sent in doc_sentences:
            for summ_sent in summary_sentences:
                premises.append(doc_sent)
                hypotheses.append(summ_sent)
        return premises, hypotheses


# ── Aggregators ───────────────────────────────────────────────────────────────

class ScoreAggregator(ABC):
    """Converts an M×N NLI matrix into one score per summary sentence (column)."""

    @abstractmethod
    def aggregate_matrix(self, nli_matrix: np.ndarray) -> List[float]:
        pass


class ZeroShotAggregator(ScoreAggregator):
    """
    SummaC-ZS: for each summary sentence, take the maximum entailment score
    across all document sentences.  No training required.

    Paper: col-wise max → vector of N scores → caller takes mean.
    """

    def aggregate_matrix(self, nli_matrix: np.ndarray) -> List[float]:
        col_maxima = nli_matrix.max(axis=0)  # shape (N_summ,)
        logger.debug("[ZS] col-max scores: %s", col_maxima.tolist())
        return col_maxima.tolist()


class ConvAggregator(ScoreAggregator):
    """
    SummaC-Conv: for each summary sentence:
      1. bin all M doc-sentence scores into H=50 histogram bins
      2. dot-product with a learned weight vector (the '1-D convolution')
      3. apply sigmoid → scalar ∈ [0,1]

    The only learned parameters are H=50 conv weights, trained via gradient
    descent on binary cross-entropy using FactCC-style synthetic data.
    """

    DEFAULT_BINS: int = 50

    def __init__(self, num_bins: int = DEFAULT_BINS):
        self._num_bins = num_bins
        self._bin_edges: np.ndarray = np.linspace(0.0, 1.0, num_bins + 1)
        self._conv_weights: Optional[np.ndarray] = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def aggregate_matrix(self, nli_matrix: np.ndarray) -> List[float]:
        if self._conv_weights is None:
            raise RuntimeError("ConvAggregator has no weights. Call .train() or .load_weights().")
        col_scores = []
        for col_idx in range(nli_matrix.shape[1]):
            col_vals = nli_matrix[:, col_idx]
            histogram = self._build_histogram(col_vals)
            score = self._apply_conv(histogram)
            logger.debug("[Conv] summ-sent %d: score=%.4f", col_idx, score)
            col_scores.append(score)
        return col_scores

    def _build_histogram(self, col_scores: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(col_scores, bins=self._bin_edges)
        return counts.astype(np.float32)

    def _apply_conv(self, histogram: np.ndarray) -> float:
        logit = float(np.dot(self._conv_weights, histogram))
        return float(1.0 / (1.0 + np.exp(-logit)))  # sigmoid

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        training_matrices: List[np.ndarray],
        training_labels: List[int],
        num_epochs: int = 100,
        learning_rate: float = 1e-2,
    ) -> None:
        """
        Train conv weights on (NLI matrix, binary label) pairs.

        Args:
            training_matrices: list of M×N NLI matrices
            training_labels:   1=consistent, 0=inconsistent
            num_epochs:        gradient descent steps
            learning_rate:     SGD step size
        """
        histograms = self._matrices_to_histograms(training_matrices)
        labels = np.array(training_labels, dtype=np.float32)
        self._conv_weights = np.zeros(self._num_bins, dtype=np.float32)
        for epoch_idx in range(num_epochs):
            epoch_loss = self._gradient_step(histograms, labels, learning_rate)
            if epoch_idx % 10 == 0:
                logger.info("[Conv train] epoch=%d loss=%.4f", epoch_idx, epoch_loss)

    def _matrices_to_histograms(self, training_matrices: List[np.ndarray]) -> np.ndarray:
        """Convert each matrix to its mean histogram across summary sentences."""
        sample_hists = []
        for matrix in training_matrices:
            col_hists = [self._build_histogram(matrix[:, j]) for j in range(matrix.shape[1])]
            sample_hists.append(np.mean(col_hists, axis=0))
        return np.stack(sample_hists)  # (N_samples, num_bins)

    def _gradient_step(
        self,
        histograms: np.ndarray,
        labels: np.ndarray,
        learning_rate: float,
    ) -> float:
        logits = histograms @ self._conv_weights
        predictions = 1.0 / (1.0 + np.exp(-logits))
        errors = predictions - labels
        gradient = (histograms.T @ errors) / len(labels)
        self._conv_weights -= learning_rate * gradient
        eps = 1e-9
        loss = float(-np.mean(
            labels * np.log(predictions + eps) + (1 - labels) * np.log(1 - predictions + eps)
        ))
        return loss

    # ── Serialization ─────────────────────────────────────────────────────────

    def save_weights(self) -> Optional[np.ndarray]:
        return self._conv_weights.copy() if self._conv_weights is not None else None

    def load_weights(self, conv_weights: np.ndarray) -> None:
        self._conv_weights = conv_weights.astype(np.float32)
        logger.info("[Conv] loaded weights shape=%s", conv_weights.shape)


# ── Pipeline ──────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    nli_scorer: NliScorer
    score_aggregator: ScoreAggregator
    decision_threshold: float = 0.5    # tuned on validation set


class SummacPipeline:
    """
    Orchestrates the full SummaC pipeline:
      1. Split source doc and summary into sentences
      2. Build M×N NLI entailment matrix
      3. Aggregate each summary-sentence column → one score per sentence
      4. Mean → final score → threshold → binary label
    """

    def __init__(self, pipeline_config: PipelineConfig):
        self._scorer = pipeline_config.nli_scorer
        self._aggregator = pipeline_config.score_aggregator
        self._threshold = pipeline_config.decision_threshold
        self._matrix_builder = PairMatrixBuilder(pipeline_config.nli_scorer)

    def score(self, source_doc: str, summary_text: str) -> SummacResult:
        logger.info("=== SummaC START ===")
        doc_sents = split_sentences(source_doc)
        summ_sents = split_sentences(summary_text)
        nli_matrix = self._matrix_builder.build(doc_sents, summ_sents)
        sent_scores = self._aggregator.aggregate_matrix(nli_matrix)
        summary_score = float(np.mean(sent_scores))
        is_consistent = summary_score >= self._threshold
        logger.info("=== SummaC END — score=%.4f consistent=%s ===", summary_score, is_consistent)
        return self._build_result(doc_sents, summ_sents, nli_matrix, sent_scores, summary_score, is_consistent)

    def _build_result(
        self,
        doc_sents: List[str],
        summ_sents: List[str],
        nli_matrix: np.ndarray,
        sent_final_scores: List[float],
        summary_score: float,
        is_consistent: bool,
    ) -> SummacResult:
        sentence_scores = [
            SentenceScore(
                sentence_text=summ_sents[j],
                raw_scores=nli_matrix[:, j].tolist(),
                final_score=sent_final_scores[j],
            )
            for j in range(len(summ_sents))
        ]
        return SummacResult(
            doc_sentences=doc_sents,
            summary_sentences=summ_sents,
            nli_matrix=nli_matrix.tolist(),
            sentence_scores=sentence_scores,
            summary_score=summary_score,
            is_consistent=is_consistent,
            threshold_used=self._threshold,
        )

    def warm_up(self) -> None:
        self._scorer.warm_up()

    def score_batch(
        self,
        doc_summary_pairs: List[Tuple[str, str]],
        max_workers: int = 1,
    ) -> List[dict]:
        """
        Run pipeline on multiple pairs (optionally in parallel).
        Returns list of dicts: {index, result, error}.
        """
        self.warm_up()

        def _score_one(pair_idx: int, source_doc: str, summary_text: str) -> dict:
            try:
                return {"index": pair_idx, "result": self.score(source_doc, summary_text), "error": None}
            except Exception:
                return {"index": pair_idx, "result": None, "error": traceback.format_exc()}

        slots: List[dict] = [{}] * len(doc_summary_pairs)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_score_one, i, doc, summ): i
                for i, (doc, summ) in enumerate(doc_summary_pairs)
            }
            for future in as_completed(futures):
                item = future.result()
                slots[item["index"]] = item
        return slots


# ── Factory ───────────────────────────────────────────────────────────────────

# ── Result serialization ──────────────────────────────────────────────────────

def serialize_result(
    sample_index: int,
    source_doc: str,
    summary_text: str,
    sample_label: str,
    summac_result: SummacResult,
) -> dict:
    """Convert a SummacResult into a JSON-serializable dict for UI storage."""
    return {
        "index": sample_index,
        "label": sample_label,
        "summary_score": round(summac_result.summary_score, 4),
        "is_consistent": summac_result.is_consistent,
        "threshold": summac_result.threshold_used,
        "doc_sentences": summac_result.doc_sentences,
        "summary_sentences": summac_result.summary_sentences,
        "nli_matrix": summac_result.nli_matrix,
        "sentence_scores": [
            {
                "text": s.sentence_text,
                "final_score": round(s.final_score, 4),
                "raw_scores": [round(v, 4) for v in s.raw_scores],
            }
            for s in summac_result.sentence_scores
        ],
        "hypothesis": summary_text,
        "premise": source_doc,
    }


_DEFAULT_NLI_MODEL = "roberta-large-mnli"


class SummacFactory:
    """Convenience factory for common pipeline configurations."""

    @staticmethod
    def zero_shot(
        model_name: str = _DEFAULT_NLI_MODEL,
        decision_threshold: float = 0.5,
    ) -> SummacPipeline:
        """SummaC-ZS: no training required."""
        config = PipelineConfig(
            nli_scorer=HfNliScorer(model_name),
            score_aggregator=ZeroShotAggregator(),
            decision_threshold=decision_threshold,
        )
        return SummacPipeline(config)

    @staticmethod
    def conv(
        model_name: str = _DEFAULT_NLI_MODEL,
        conv_weights: Optional[np.ndarray] = None,
        decision_threshold: float = 0.5,
    ) -> SummacPipeline:
        """SummaC-Conv: requires trained weights or call .train() on the aggregator."""
        aggregator = ConvAggregator()
        if conv_weights is not None:
            aggregator.load_weights(conv_weights)
        config = PipelineConfig(
            nli_scorer=HfNliScorer(model_name),
            score_aggregator=aggregator,
            decision_threshold=decision_threshold,
        )
        return SummacPipeline(config)
