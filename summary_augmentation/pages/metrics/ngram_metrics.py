"""
N-gram based evaluation metrics for Hebrew NLI summarization evaluation.

Provides:
    RougeCalculator      : ROUGE-N and ROUGE-L recall-oriented scores
    BleuCalculator       : BLEU precision-oriented score with brevity penalty
    ChrfCalculator       : ChrF character n-gram F-score (suited for Hebrew morphology)
    NovelNgramCalculator : Fraction of summary n-grams absent from the source document
                          (abstractiveness / hallucination-risk proxy, per Aharoni et al. 2023)
"""
from typing import List, Dict, Tuple
from collections import Counter
import math


# ─── Shared tokenization helpers ─────────────────────────────────────────────

def _tokenize_to_words(text: str) -> List[str]:
    return text.strip().split()


def _extract_word_ngrams(word_tokens: List[str], ngram_size: int) -> Counter:
    ngrams = [tuple(word_tokens[i:i + ngram_size]) for i in range(len(word_tokens) - ngram_size + 1)]
    return Counter(ngrams)


def _extract_character_ngrams(text: str, ngram_size: int) -> Counter:
    clean_text = text.replace(' ', '')
    ngrams = [clean_text[i:i + ngram_size] for i in range(len(clean_text) - ngram_size + 1)]
    return Counter(ngrams)


# ─── Shared scoring helper ────────────────────────────────────────────────────

def _compute_precision_recall_f1(overlap_count: int, reference_count: int, hypothesis_count: int) -> Dict[str, float]:
    precision = overlap_count / hypothesis_count if hypothesis_count > 0 else 0.0
    recall = overlap_count / reference_count if reference_count > 0 else 0.0
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator > 0 else 0.0
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }


# ─── LCS builder (used internally by RougeCalculator) ────────────────────────

class _LcsBuilder:
    """Builds the LCS dynamic programming table for two token sequences."""

    def __init__(self, reference_tokens: List[str], hypothesis_tokens: List[str]):
        self.reference_tokens = reference_tokens
        self.hypothesis_tokens = hypothesis_tokens
        self.table = self._build_table()

    def _build_table(self) -> List[List[int]]:
        reference_length = len(self.reference_tokens)
        hypothesis_length = len(self.hypothesis_tokens)
        table = [[0] * (hypothesis_length + 1) for _ in range(reference_length + 1)]
        for row in range(1, reference_length + 1):
            for col in range(1, hypothesis_length + 1):
                table[row][col] = self._compute_cell_value(table, row, col)
        return table

    def _compute_cell_value(self, dp_table: List[List[int]], reference_position: int, hypothesis_position: int) -> int:
        is_match = self.reference_tokens[reference_position - 1] == self.hypothesis_tokens[hypothesis_position - 1]
        if is_match:
            return dp_table[reference_position - 1][hypothesis_position - 1] + 1
        return max(dp_table[reference_position - 1][hypothesis_position], dp_table[reference_position][hypothesis_position - 1])

    def get_lcs_length(self) -> int:
        return self.table[len(self.reference_tokens)][len(self.hypothesis_tokens)]


# ─── ROUGE ────────────────────────────────────────────────────────────────────

class RougeCalculator:
    """Computes ROUGE-N and ROUGE-L recall-oriented n-gram overlap scores.

    Usage:
        calculator = RougeCalculator(reference, hypothesis)
        rouge1 = calculator.compute_rouge_n(1)
        rouge_l = calculator.compute_rouge_l()
    """

    def __init__(self, reference: str, hypothesis: str):
        self.reference_tokens = _tokenize_to_words(reference)
        self.hypothesis_tokens = _tokenize_to_words(hypothesis)

    def compute_rouge_n(self, ngram_size: int) -> Dict[str, float]:
        if len(self.reference_tokens) < ngram_size or len(self.hypothesis_tokens) < ngram_size:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        reference_ngrams = _extract_word_ngrams(self.reference_tokens, ngram_size)
        hypothesis_ngrams = _extract_word_ngrams(self.hypothesis_tokens, ngram_size)
        overlap_count = sum((reference_ngrams & hypothesis_ngrams).values())
        return _compute_precision_recall_f1(
            overlap_count,
            sum(reference_ngrams.values()),
            sum(hypothesis_ngrams.values())
        )

    def compute_rouge_l(self) -> Dict[str, float]:
        lcs_length = _LcsBuilder(self.reference_tokens, self.hypothesis_tokens).get_lcs_length()
        return _compute_precision_recall_f1(
            lcs_length,
            len(self.reference_tokens),
            len(self.hypothesis_tokens)
        )


# ─── BLEU ─────────────────────────────────────────────────────────────────────

class BleuCalculator:
    """Computes BLEU precision-oriented n-gram score with brevity penalty.

    Usage:
        calculator = BleuCalculator(reference, hypothesis)
        result = calculator.compute_bleu(maximum_ngram_order=4)
    """

    def __init__(self, reference: str, hypothesis: str):
        self.reference_tokens = _tokenize_to_words(reference)
        self.hypothesis_tokens = _tokenize_to_words(hypothesis)

    def compute_bleu(self, maximum_ngram_order: int = 4) -> Dict:
        ngram_precisions = self._compute_all_ngram_precisions(maximum_ngram_order)
        brevity_penalty = self._compute_brevity_penalty()
        geometric_mean = self._compute_geometric_mean(ngram_precisions)
        bleu_score = brevity_penalty * geometric_mean
        return {
            'bleu': round(bleu_score, 4),
            'brevity_penalty': round(brevity_penalty, 4),
            'scores': {f'bleu-{n + 1}': round(ngram_precisions[n], 4) for n in range(len(ngram_precisions))}
        }

    def _compute_ngram_precision(self, ngram_size: int) -> float:
        if len(self.hypothesis_tokens) < ngram_size or len(self.reference_tokens) < ngram_size:
            return 0.0
        reference_ngrams = _extract_word_ngrams(self.reference_tokens, ngram_size)
        hypothesis_ngrams = _extract_word_ngrams(self.hypothesis_tokens, ngram_size)
        overlap_count = sum((reference_ngrams & hypothesis_ngrams).values())
        total_hypothesis_ngrams = sum(hypothesis_ngrams.values())
        return overlap_count / total_hypothesis_ngrams if total_hypothesis_ngrams > 0 else 0.0

    def _compute_all_ngram_precisions(self, maximum_ngram_order: int) -> List[float]:
        return [self._compute_ngram_precision(ngram_size) for ngram_size in range(1, maximum_ngram_order + 1)]

    def _compute_brevity_penalty(self) -> float:
        hypothesis_length = len(self.hypothesis_tokens)
        reference_length = len(self.reference_tokens)
        if hypothesis_length == 0 or reference_length == 0:
            return 0.0
        if hypothesis_length >= reference_length:
            return 1.0
        return math.exp(1 - reference_length / hypothesis_length)

    @staticmethod
    def _compute_geometric_mean(precisions: List[float]) -> float:
        if any(precision <= 0 for precision in precisions):
            return 0.0
        log_sum = sum(math.log(precision) for precision in precisions)
        return math.exp(log_sum / len(precisions))


# ─── Token F1 ────────────────────────────────────────────────────────────────

class TokenF1Calculator:
    """
    Token-level F1 using set intersection (each unique word counted once).

    Unlike ROUGE-1 which uses Counter intersection (clipped by frequency),
    this treats each word type as either present or absent — matching the
    SQuAD / TRUE paper baseline definition.

    Usage:
        result = TokenF1Calculator(reference, hypothesis).compute()
        # {'precision': ..., 'recall': ..., 'f1': ...}
    """

    def __init__(self, reference: str, hypothesis: str):
        self.reference_tokens = set(_tokenize_to_words(reference))
        self.hypothesis_tokens = set(_tokenize_to_words(hypothesis))

    def compute(self) -> Dict[str, float]:
        if not self.reference_tokens or not self.hypothesis_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        overlap = len(self.reference_tokens & self.hypothesis_tokens)
        return _compute_precision_recall_f1(overlap, len(self.reference_tokens), len(self.hypothesis_tokens))


# ─── ChrF ─────────────────────────────────────────────────────────────────────

class ChrfCalculator:
    """Computes ChrF character n-gram F-score, suited for morphologically rich languages like Hebrew.

    Usage:
        calculator = ChrfCalculator(reference, hypothesis)
        result = calculator.compute_chrf(maximum_char_ngram=6)
    """

    def __init__(self, reference: str, hypothesis: str):
        self.reference = reference
        self.hypothesis = hypothesis

    def compute_chrf(self, maximum_char_ngram: int = 6, beta: int = 2) -> Dict:
        total_precision, total_recall = self._sum_precision_recall_across_ngrams(maximum_char_ngram)
        average_precision = total_precision / maximum_char_ngram
        average_recall = total_recall / maximum_char_ngram
        chrf_score = self._apply_beta_weighted_f1(average_precision, average_recall, beta)
        return {
            'chrf': round(chrf_score * 100, 2),
            'precision': round(average_precision, 4),
            'recall': round(average_recall, 4)
        }

    def _compute_single_ngram_precision_recall(self, ngram_size: int) -> Tuple[float, float]:
        reference_ngrams = _extract_character_ngrams(self.reference, ngram_size)
        hypothesis_ngrams = _extract_character_ngrams(self.hypothesis, ngram_size)
        reference_total = sum(reference_ngrams.values())
        hypothesis_total = sum(hypothesis_ngrams.values())
        if reference_total == 0 or hypothesis_total == 0:
            return 0.0, 0.0
        overlap_count = sum((reference_ngrams & hypothesis_ngrams).values())
        return overlap_count / hypothesis_total, overlap_count / reference_total

    def _sum_precision_recall_across_ngrams(self, maximum_char_ngram: int) -> Tuple[float, float]:
        total_precision = 0.0
        total_recall = 0.0
        for ngram_size in range(1, maximum_char_ngram + 1):
            precision, recall = self._compute_single_ngram_precision_recall(ngram_size)
            total_precision += precision
            total_recall += recall
        return total_precision, total_recall

    @staticmethod
    def _apply_beta_weighted_f1(average_precision: float, average_recall: float, beta: int) -> float:
        if average_precision + average_recall == 0:
            return 0.0
        beta_squared = beta ** 2
        numerator = (1 + beta_squared) * average_precision * average_recall
        denominator = (beta_squared * average_precision) + average_recall
        return numerator / denominator


# ─── Novel N-gram ─────────────────────────────────────────────────────────────

class NovelNgramCalculator:
    """Fraction of summary n-grams that do not appear in the source document.

    Measures abstractiveness: a higher value means the summary introduces
    more content not grounded in the source, which correlates with hallucination
    risk (Aharoni et al., 2023 — Detecting and Mitigating Hallucinations in
    Multilingual Summarisation).

    Computed as:
        novel_n% = Σ count(ngram) for novel ngrams / Σ count(ngram) for all summary ngrams

    where "novel" means the n-gram tuple does not appear anywhere in the source.
    Counts are frequency-weighted (Counter-based), so a repeated novel bigram
    contributes proportionally more than a single occurrence.

    Usage:
        calc = NovelNgramCalculator(source_document, summary_text)
        result = calc.compute()
        # {'novel_unigram_pct': 0.12, 'novel_bigram_pct': 0.34, 'novel_trigram_pct': 0.51}
    """

    def __init__(
        self,
        source_document: str,
        summary_text: str,
    ) -> None:
        self._source_tokens = _tokenize_to_words(source_document)
        self._summary_tokens = _tokenize_to_words(summary_text)

    def compute_novel_ngram_pct(
        self,
        ngram_size: int,
    ) -> float:
        """Return the fraction (0.0–1.0) of summary n-grams absent from the source."""
        if len(self._summary_tokens) < ngram_size:
            return 0.0
        source_ngram_types = set(_extract_word_ngrams(self._source_tokens, ngram_size).keys())
        summary_ngram_counts = _extract_word_ngrams(self._summary_tokens, ngram_size)
        total_summary_ngram_count = sum(summary_ngram_counts.values())
        if total_summary_ngram_count == 0:
            return 0.0
        novel_ngram_count = sum(
            ngram_count
            for ngram_tuple, ngram_count in summary_ngram_counts.items()
            if ngram_tuple not in source_ngram_types
        )
        return round(novel_ngram_count / total_summary_ngram_count, 4)

    def compute(self) -> Dict[str, float]:
        """Return novel n-gram percentages for unigrams, bigrams, and trigrams."""
        return {
            'novel_unigram_pct': self.compute_novel_ngram_pct(1),
            'novel_bigram_pct':  self.compute_novel_ngram_pct(2),
            'novel_trigram_pct': self.compute_novel_ngram_pct(3),
        }
