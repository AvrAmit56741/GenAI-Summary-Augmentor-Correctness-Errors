"""
Evaluation metric comparison backend.

Provides statistical tests to measure how well a scoring metric
(e.g. ROUGE, BLEU, model confidence) predicts binary correctness labels.

Classes:
    ComparisonResult       : Immutable result object shared by all calculators
    KendallTauCalculator   : Rank correlation between metric scores and labels
    PointBiserialCalculator: Correlation between continuous scores and binary labels
"""
from dataclasses import dataclass
from typing import List
from scipy.stats import kendalltau, pointbiserialr
import math


@dataclass(frozen=True)
class ComparisonResult:
    """Shared result object returned by all comparison calculators."""
    metric_name: str
    statistic: float       # τ for Kendall, r for Point-biserial
    p_value: float
    is_significant: bool   # p < 0.05
    interpretation: str    # human-readable verdict


def _interpret_strength(statistic: float) -> str:
    absolute = abs(statistic)
    if math.isnan(absolute):
        return "Undefined — no label variance or metric is constant"
    if absolute > 0.5:
        return "Strong correlation"
    if absolute > 0.3:
        return "Moderate correlation"
    if absolute > 0.1:
        return "Weak correlation"
    return "No meaningful correlation"


def _build_result(metric_name: str, statistic: float, p_value: float) -> ComparisonResult:
    is_significant = (not math.isnan(p_value)) and (p_value < 0.05)
    interpretation = _interpret_strength(statistic)
    return ComparisonResult(
        metric_name=metric_name,
        statistic=statistic,
        p_value=p_value,
        is_significant=is_significant,
        interpretation=interpretation,
    )


# ─── Kendall τ ────────────────────────────────────────────────────────────────

class KendallTauCalculator:
    """Rank correlation between a metric score column and binary correctness labels.

    Asks: do higher metric scores tend to appear on correct (label=1) summaries?

    Usage:
        result = KendallTauCalculator(scores, labels, "ROUGE-1 F1").compute()
    """

    def __init__(self, metric_scores: List[float], binary_labels: List[int], metric_name: str):
        self.metric_scores = metric_scores
        self.binary_labels = binary_labels
        self.metric_name = metric_name

    def compute(self) -> ComparisonResult:
        tau, p_value = kendalltau(self.metric_scores, self.binary_labels)
        return _build_result(self.metric_name, tau, p_value)


# ─── Point-biserial ───────────────────────────────────────────────────────────

class PointBiserialCalculator:
    """Correlation between a continuous metric score and a binary label.

    Statistically more appropriate than Kendall τ when one variable is binary.
    Equivalent to Pearson r applied to a binary grouping variable.

    Usage:
        result = PointBiserialCalculator(scores, labels, "BLEU").compute()
    """

    def __init__(self, metric_scores: List[float], binary_labels: List[int], metric_name: str):
        self.metric_scores = metric_scores
        self.binary_labels = binary_labels
        self.metric_name = metric_name

    def compute(self) -> ComparisonResult:
        correlation, p_value = pointbiserialr(self.binary_labels, self.metric_scores)
        return _build_result(self.metric_name, correlation, p_value)
