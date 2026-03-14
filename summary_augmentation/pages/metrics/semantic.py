"""
Semantic similarity metrics using contextual embeddings.

Provides:
    BertScoreCalculator : BERTScore using Hebrew BERT models
    BERTSCORE_AVAILABLE : bool flag indicating whether the bert-score library is installed
"""
from typing import Dict, Any

# Optional import — bert-score is a heavy optional dependency
try:
    from bert_score import score as _bert_score_function
    BERTSCORE_AVAILABLE = True
except ImportError:
    _bert_score_function = None
    BERTSCORE_AVAILABLE = False


class BertScoreCalculator:
    """Computes BERTScore semantic similarity using Hebrew BERT contextual embeddings.

    Usage:
        calculator = BertScoreCalculator(reference, hypothesis, model_name)
        result = calculator.compute_bert_score()
    """

    def __init__(self, reference: str, hypothesis: str, model_name: str):
        self.reference = reference
        self.hypothesis = hypothesis
        self.model_name = model_name

    def compute_bert_score(self) -> Dict[str, Any]:
        if not BERTSCORE_AVAILABLE:
            return self._build_library_not_installed_error()
        try:
            return self._run_bert_score()
        except Exception as exception:
            return self._build_exception_error(exception)

    def _run_bert_score(self) -> Dict[str, Any]:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision, recall, f1 = _bert_score_function(
            cands=[self.hypothesis],
            refs=[self.reference],
            model_type=self.model_name,
            num_layers=12,
            lang="he",
            rescale_with_baseline=False,
            verbose=False,
            device=device,
            batch_size=1
        )
        return {
            'precision': round(precision.mean().item(), 4),
            'recall': round(recall.mean().item(), 4),
            'f1': round(f1.mean().item(), 4)
        }

    def _build_exception_error(self, exception: Exception) -> Dict[str, Any]:
        error_message = str(exception)
        is_model_not_recognized = "model_type" in error_message or "not found" in error_message.lower()
        if is_model_not_recognized:
            return self._build_model_not_recognized_error()
        return self._build_generic_error(exception)

    @staticmethod
    def _build_library_not_installed_error() -> Dict[str, Any]:
        return {
            'error': "bert_score library not installed. Run: pip install bert-score",
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    def _build_model_not_recognized_error(self) -> Dict[str, Any]:
        return {
            'error': f"Model '{self.model_name}' not recognized by bert_score. Try 'avichr/heBERT' instead.",
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    @staticmethod
    def _build_generic_error(exception: Exception) -> Dict[str, Any]:
        return {
            'error': f"{type(exception).__name__}: {str(exception)}",
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
