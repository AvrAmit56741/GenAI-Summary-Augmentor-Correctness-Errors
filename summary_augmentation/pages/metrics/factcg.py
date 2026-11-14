"""
FactCG factual-consistency scorer (Lei et al., NAACL 2025).

Port of the reference inference in github.com/derenlei/FactCG (factcg/inference.py, InferenceForCG):
    * model      : yaxili96/FactCG-DeBERTa-v3-Large (DeBERTa-v3-large, English, MIT)
    * input      : instruction template with the document chunk and the claim (summary)
    * chunking   : document split into sentence-aligned chunks of <= 550 tokens
    * score      : P(label 1 = supported), maximised over document chunks
Differences from the reference: sentences are split with a regex and tokens counted by whitespace
instead of NLTK punkt / word_tokenize (no NLTK download needed; works for Hebrew punctuation).

Usage:
    scorer = FactCGScorer()
    result = scorer.score(source_document, summary)      # -> dict(score, num_chunks, best_chunk_index)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

MODEL_NAME = "yaxili96/FactCG-DeBERTa-v3-Large"
MAX_CHUNK_TOKENS = 550
MAX_INPUT_TOKENS = 2048
INSTRUCTION_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that \"{text_b}\"?"
    "\n\nOPTIONS:\n- Yes\n- No\nI think the answer is "
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class FactCGResult:
    score: float               # P(supported), max over chunks; higher = more faithful
    num_chunks: int
    best_chunk_index: int


def chunk_document(text: str, max_chunk_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """Greedy sentence packing into chunks of at most `max_chunk_tokens` whitespace tokens."""
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]
    chunks: List[str] = []
    current, current_len = [], 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and current_len + n > max_chunk_tokens:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]


class FactCGScorer:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None, batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._tokenizer = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        return self._device

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        config = AutoConfig.from_pretrained(self.model_name, num_labels=2, finetuning_task="text-classification")
        config.problem_type = "single_label_classification"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
        self._model.to(self.device).eval()

    def is_loaded(self) -> bool:
        return self._model is not None

    def _support_probabilities(self, documents: List[str], claims: List[str]) -> List[float]:
        import torch
        texts = [INSTRUCTION_TEMPLATE.format(text_a=d, text_b=c) for d, c in zip(documents, claims)]
        probs: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self._tokenizer(texts[i:i + self.batch_size], max_length=MAX_INPUT_TOKENS, truncation=True,
                                  padding="longest", return_tensors="pt").to(self.device)
            with torch.no_grad():
                p = torch.softmax(self._model(**enc).logits.float(), dim=-1)[:, 1]
            probs.extend(p.cpu().tolist())
        return probs

    def score(self, source_document: str, summary: str) -> FactCGResult:
        self.load()
        chunks = chunk_document(source_document)
        probs = self._support_probabilities(chunks, [summary] * len(chunks))
        best = max(range(len(probs)), key=probs.__getitem__)
        return FactCGResult(score=float(probs[best]), num_chunks=len(chunks), best_chunk_index=best)
