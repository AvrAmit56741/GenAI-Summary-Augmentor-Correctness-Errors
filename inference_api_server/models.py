"""
BARTScore and BERTScore model loading and inference.
Loaded once at server startup; called from server_api.py endpoints.
"""
from __future__ import annotations

import logging
import math

import torch
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

BARTSCORE_MODEL_NAME = "facebook/bart-large-cnn"
BARTSCORE_MAX_SOURCE_LENGTH = 1024
BARTSCORE_MAX_SUMMARY_LENGTH = 128

BERTSCORE_MODEL_NAME = "avichr/heBERT"
BERTSCORE_NUM_LAYERS = 12

# ── Global state ──────────────────────────────────────────────────────────────

_bartscore_model: BartForConditionalGeneration | None = None
_bartscore_tokenizer: BartTokenizer | None = None
_bartscore_device: str | None = None

_bertscore_ready: bool = False

# SummaC — keyed by model_name so multiple NLI models can coexist
_summac_pipelines: dict = {}
_summac_available: bool = False


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_bartscore() -> None:
    global _bartscore_model, _bartscore_tokenizer, _bartscore_device
    logger.info("Loading BARTScore model %s ...", BARTSCORE_MODEL_NAME)
    try:
        _bartscore_tokenizer = BartTokenizer.from_pretrained(BARTSCORE_MODEL_NAME)
        _bartscore_model = BartForConditionalGeneration.from_pretrained(BARTSCORE_MODEL_NAME)
        _bartscore_device = "cuda" if torch.cuda.is_available() else "cpu"
        _bartscore_model.to(_bartscore_device)
        _bartscore_model.eval()
        logger.info("✅ BARTScore loaded on %s", _bartscore_device)
    except Exception as exc:
        logger.error("❌ Failed to load BARTScore: %s", exc)


def load_bertscore() -> None:
    global _bertscore_ready
    logger.info("Checking bert-score library for model %s ...", BERTSCORE_MODEL_NAME)
    try:
        from bert_score import score as _bert_score_fn  # noqa: F401
        _bertscore_ready = True
        logger.info("✅ BERTScore ready (model: %s)", BERTSCORE_MODEL_NAME)
    except ImportError:
        logger.error("❌ bert_score not installed — run: pip install bert-score")


# ── Status helpers ────────────────────────────────────────────────────────────

def bartscore_is_loaded() -> bool:
    return _bartscore_model is not None


def bertscore_is_ready() -> bool:
    return _bertscore_ready


def load_summac() -> None:
    global _summac_available
    logger.info("Checking SummaC dependencies ...")
    try:
        from transformers import pipeline as _hf_pipeline  # noqa: F401
        _summac_available = True
        logger.info("✅ SummaC ready (NLI models load on first request per model)")
    except ImportError:
        logger.error("❌ transformers not installed — SummaC unavailable")


def summac_is_ready() -> bool:
    return _summac_available


# ── Inference ─────────────────────────────────────────────────────────────────

def compute_bartscore(source_document: str, summary: str) -> dict:
    """
    Returns avg log-prob per token (raw_score) and exp(raw_score) as normalized_score.
    Raises RuntimeError if model not loaded.
    """
    if _bartscore_model is None or _bartscore_tokenizer is None:
        raise RuntimeError("BARTScore model not loaded")

    source_enc = _bartscore_tokenizer(
        source_document.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=BARTSCORE_MAX_SOURCE_LENGTH,
        padding=True,
    )
    target_enc = _bartscore_tokenizer(
        summary.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=BARTSCORE_MAX_SUMMARY_LENGTH,
        padding=True,
    )

    input_ids = source_enc["input_ids"].to(_bartscore_device)
    attention_mask = source_enc["attention_mask"].to(_bartscore_device)
    target_labels = target_enc["input_ids"].clone().to(_bartscore_device)
    target_labels[target_labels == _bartscore_tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = _bartscore_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_labels,
        )

    batch_size, _, vocab_size = outputs.logits.shape
    loss_fn = CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token_loss = loss_fn(
        outputs.logits.view(-1, vocab_size),
        target_labels.view(-1),
    ).view(batch_size, -1)

    token_counts = (target_labels != -100).sum(dim=1).float().clamp(min=1)
    raw_score = float((-per_token_loss.sum(dim=1) / token_counts).cpu().item())
    normalized_score = math.exp(raw_score)

    return {
        "raw_score": raw_score,
        "normalized_score": normalized_score,
        "model_name": BARTSCORE_MODEL_NAME,
    }


def compute_bertscore(reference: str, hypothesis: str, model_name: str = BERTSCORE_MODEL_NAME) -> dict:
    """
    Returns BERTScore precision, recall, F1.
    model_name is passed through to bert_score — different models are loaded on first use
    and cached internally by the bert_score library for subsequent calls.
    Raises RuntimeError if library not available.
    """
    if not _bertscore_ready:
        raise RuntimeError("BERTScore not available — bert_score library missing")

    from bert_score import score as bert_score_fn

    # BERT models have a 512-token limit; truncate by words to stay within it
    _MAX_WORDS = 350
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    ref_text = " ".join(ref_words[:_MAX_WORDS]) if len(ref_words) > _MAX_WORDS else reference.strip()
    hyp_text = " ".join(hyp_words[:_MAX_WORDS]) if len(hyp_words) > _MAX_WORDS else hypothesis.strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision, recall, f1 = bert_score_fn(
        cands=[hyp_text],
        refs=[ref_text],
        model_type=model_name,
        num_layers=BERTSCORE_NUM_LAYERS,
        lang="he",
        rescale_with_baseline=False,
        verbose=False,
        device=device,
        batch_size=1,
        trust_remote_code=True,
    )
    return {
        "precision": round(float(precision.mean().item()), 4),
        "recall": round(float(recall.mean().item()), 4),
        "f1": round(float(f1.mean().item()), 4),
        "model_name": model_name,
    }


def compute_summac(source_doc: str, summary: str, model_name: str = "roberta-large-mnli") -> dict:
    """
    Run SummaC-ZS pipeline: sentence-split → NLI pair matrix → col-max → mean.
    Returns raw scores only; threshold/consistency decision is the caller's responsibility.
    NLI models are cached by model_name after first load.
    """
    if not _summac_available:
        raise RuntimeError("SummaC not available — transformers not installed")

    import os
    import sys
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from summary_augmentation.pages.metrics.summac import SummacFactory

    if model_name not in _summac_pipelines:
        logger.info("Loading SummaC NLI model: %s", model_name)
        pipeline = SummacFactory.zero_shot(model_name, decision_threshold=0.5)
        pipeline.warm_up()
        _summac_pipelines[model_name] = pipeline
        logger.info("✅ SummaC NLI model loaded: %s", model_name)

    result = _summac_pipelines[model_name].score(source_doc.strip(), summary.strip())

    return {
        "summary_score": round(result.summary_score, 4),
        "doc_sentences": result.doc_sentences,
        "summary_sentences": result.summary_sentences,
        "nli_matrix": result.nli_matrix,
        "sentence_scores": [
            {
                "text": s.sentence_text,
                "final_score": round(s.final_score, 4),
                "raw_scores": [round(v, 4) for v in s.raw_scores],
            }
            for s in result.sentence_scores
        ],
        "model_name": model_name,
    }
