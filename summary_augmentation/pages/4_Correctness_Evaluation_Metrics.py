import sys
import os
import json
import streamlit as st
from typing import Dict, List, Any, Optional
from collections import Counter
import re
from scipy.stats import kendalltau

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import bert_score (optional dependency)
try:
    from bert_score import score as bert_score_func

    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    bert_score_func = None

st.set_page_config(page_title="Correctness Evaluation Metrics", layout="wide", page_icon="📏")

st.title("📏 Correctness Evaluation Metrics")
st.markdown(
    '<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>',
    unsafe_allow_html=True)

st.markdown("""
Evaluate summary quality using standard text-overlap metrics. Compare a generated summary against its source document.
""")

# Create tabs for single vs batch evaluation
tab_single, tab_batch, tab_classifier = st.tabs(["Single Evaluation", "Batch Evaluation", "Classifier Predictions"])


# ==================== METRIC COMPUTATION FUNCTIONS ====================

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization"""
    return text.strip().split()


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token list"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i + n]))
    return Counter(ngrams)


def get_char_ngrams(text: str, n: int) -> Counter:
    """Extract character n-grams"""
    chars = text.replace(' ', '')  # Remove spaces for character-level
    ngrams = []
    for i in range(len(chars) - n + 1):
        ngrams.append(chars[i:i + n])
    return Counter(ngrams)


def compute_rouge_n(reference: str, hypothesis: str, n: int) -> Dict[str, float]:
    """
    Compute ROUGE-N score
    ROUGE-N: Recall-oriented measure of n-gram overlap
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if len(ref_tokens) < n or len(hyp_tokens) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_ngrams = get_ngrams(ref_tokens, n)
    hyp_ngrams = get_ngrams(hyp_tokens, n)

    # Count overlapping n-grams
    overlap = sum((ref_ngrams & hyp_ngrams).values())

    # Precision: overlap / hypothesis n-grams
    precision = overlap / sum(hyp_ngrams.values()) if sum(hyp_ngrams.values()) > 0 else 0.0

    # Recall: overlap / reference n-grams (ROUGE emphasis)
    recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0.0

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def lcs_length(x: List[str], y: List[str]) -> int:
    """Compute Longest Common Subsequence length"""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def compute_rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-L score
    ROUGE-L: Longest Common Subsequence based measure
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = lcs_length(ref_tokens, hyp_tokens)

    precision = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
    recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
    """
    Compute BLEU score
    BLEU: Precision-oriented n-gram overlap with brevity penalty
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if len(hyp_tokens) == 0:
        return {"bleu": 0.0, "brevity_penalty": 0.0, "scores": {}}

    # Compute precision for each n-gram order
    precisions: List[float] = []
    scores_detail: Dict[str, float] = {}

    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n or len(ref_tokens) < n:
            precisions.append(0.0)
            scores_detail[f"bleu-{n}"] = 0.0
            continue

        ref_ngrams = get_ngrams(ref_tokens, n)
        hyp_ngrams = get_ngrams(hyp_tokens, n)

        overlap = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())

        prec = overlap / total if total > 0 else 0.0
        precisions.append(prec)
        scores_detail[f"bleu-{n}"] = round(prec, 4)

    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = 1.0
        for p in precisions:
            geo_mean *= p
        geo_mean = geo_mean ** (1.0 / len(precisions))
    else:
        geo_mean = 0.0

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = min(1.0, (len(hyp_tokens) / len(ref_tokens)) ** 0.5) if len(ref_tokens) > 0 else 0.0

    bleu_score = bp * geo_mean

    return {
        "bleu": round(bleu_score, 4),
        "brevity_penalty": round(bp, 4),
        "scores": scores_detail
    }


def compute_chrf(reference: str, hypothesis: str, n: int = 6, beta: int = 2) -> Dict[str, float]:
    """
    Compute ChrF / ChrF++ score
    ChrF: Character n-gram F-score (good for morphologically rich languages)
    """
    if len(reference) == 0 or len(hypothesis) == 0:
        return {"chrf": 0.0, "precision": 0.0, "recall": 0.0}

    # Character-level n-gram matching
    total_precision = 0.0
    total_recall = 0.0

    for i in range(1, n + 1):
        ref_ngrams = get_char_ngrams(reference, i)
        hyp_ngrams = get_char_ngrams(hypothesis, i)

        if sum(hyp_ngrams.values()) == 0 or sum(ref_ngrams.values()) == 0:
            continue

        overlap = sum((ref_ngrams & hyp_ngrams).values())

        prec = overlap / sum(hyp_ngrams.values()) if sum(hyp_ngrams.values()) > 0 else 0.0
        rec = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0.0

        total_precision += prec
        total_recall += rec

    avg_precision = total_precision / n if n > 0 else 0.0
    avg_recall = total_recall / n if n > 0 else 0.0

    # ChrF uses beta=2 by default (favors recall)
    if avg_precision + avg_recall > 0:
        chrf_score = (1 + beta ** 2) * (avg_precision * avg_recall) / ((beta ** 2 * avg_precision) + avg_recall)
    else:
        chrf_score = 0.0

    return {
        "chrf": round(chrf_score * 100, 2),  # ChrF is typically reported as percentage
        "precision": round(avg_precision, 4),
        "recall": round(avg_recall, 4)
    }


def compute_bertscore(reference: str, hypothesis: str, model_name: str) -> Dict[str, Any]:
    """
    Compute BERTScore using contextual embeddings
    BERTScore: Semantic similarity based on BERT embeddings
    
    Args:
        reference: Source/reference text
        hypothesis: Generated summary
        model_name: Hebrew BERT model to use
    
    Returns:
        Dictionary with precision, recall, f1 scores
    """
    if not BERTSCORE_AVAILABLE:
        return {
            "error": "bert_score library not installed. Run: pip install bert-score",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    try:
        import torch

        # For bert_score, we need to use the model directly if not in registry
        # All Hebrew BERT models should work with model_type parameter
        P, R, F1 = bert_score_func(
            cands=[hypothesis],
            refs=[reference],
            model_type=model_name,
            num_layers=12,  # Use all layers for BERT-base models
            lang="he",  # Hebrew tokenization
            rescale_with_baseline=False,  # Raw F1 scores
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=1
        )

        return {
            "precision": round(P.mean().item(), 4),
            "recall": round(R.mean().item(), 4),
            "f1": round(F1.mean().item(), 4)
        }
    except Exception as e:
        # If still failing, provide helpful error message
        error_msg = str(e)
        if "model_type" in error_msg or "not found" in error_msg.lower():
            return {
                "error": f"Model '{model_name}' not recognized by bert_score. Try 'avichr/heBERT' instead.",
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        return {
            "error": f"{type(e).__name__}: {str(e)}",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }


# ==================== UI LAYOUT ====================

# ===== SINGLE EVALUATION TAB =====
with tab_single:
    st.markdown("---")

# Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Source Document (Reference)")
    source_input_method = st.radio(
        "Input method:",
        ["Paste Text", "Upload File"],
        key="source_method",
        horizontal=True
    )

    if source_input_method == "Paste Text":
        source_text = st.text_area(
            "Enter source document:",
            height=300,
            placeholder="Paste the source article/document here...",
            key="source_text"
        )
    else:
        source_file = st.file_uploader(
            "Upload source file:",
            type=["txt"],
            key="source_file"
        )
        if source_file:
            source_text = source_file.read().decode('utf-8')
            st.text_area("Source preview:", value=source_text, height=200, disabled=True)
        else:
            source_text = ""

with col2:
    st.subheader("📝 Generated Summary (Hypothesis)")
    summary_input_method = st.radio(
        "Input method:",
        ["Paste Text", "Upload File"],
        key="summary_method",
        horizontal=True
    )

    if summary_input_method == "Paste Text":
        summary_text = st.text_area(
            "Enter generated summary:",
            height=300,
            placeholder="Paste the generated summary here...",
            key="summary_text"
        )
    else:
        summary_file = st.file_uploader(
            "Upload summary file:",
            type=["txt"],
            key="summary_file"
        )
        if summary_file:
            summary_text = summary_file.read().decode('utf-8')
            st.text_area("Summary preview:", value=summary_text, height=200, disabled=True)
        else:
            summary_text = ""

st.markdown("---")

# Metric selection
st.subheader("📊 Select Metrics to Compute")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Surface-Level Metrics** (no model required)")
    compute_rouge = st.checkbox("ROUGE (N-gram Recall)", value=True, key="rouge_check")
    if compute_rouge:
        rouge_variants = st.multiselect(
            "ROUGE variants:",
            ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            default=["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            key="rouge_variants"
        )

    compute_bleu_check = st.checkbox("BLEU (N-gram Precision)", value=True, key="bleu_check")
    if compute_bleu_check:
        bleu_max_n = st.slider(
            "Max N-gram order:",
            min_value=1,
            max_value=4,
            value=4,
            key="bleu_n"
        )

    compute_chrf_check = st.checkbox("ChrF (Character N-grams)", value=True, key="chrf_check")
    if compute_chrf_check:
        chrf_max_n = st.slider(
            "Max character N-gram:",
            min_value=2,
            max_value=6,
            value=6,
            key="chrf_n"
        )

with col2:
    st.markdown("**Semantic Metrics** (requires Hebrew BERT)")

    if not BERTSCORE_AVAILABLE:
        st.warning("⚠️ BERTScore not available. Install: `pip install bert-score`")
        compute_bertscore_check = False
    else:
        compute_bertscore_check = st.checkbox("BERTScore (Semantic Similarity)", value=False, key="bertscore_check")

    if compute_bertscore_check:
        bert_model = st.selectbox(
            "Hebrew BERT Model:",
            [
                "avichr/heBERT",
                "onlplab/alephbert-base",
                "dicta-il/neodictabert"
            ],
            index=0,
            key="bert_model",
            help="Choose Hebrew BERT model for semantic similarity"
        )

        with st.expander("ℹ️ Model Information"):
            st.markdown("""
            **avichr/heBERT** ✅ (Recommended): Trained on OSCAR + Wikipedia + UGC emotion data  
            **onlplab/alephbert-base**: AlephBERT from Open University NLP Lab  
            **dicta-il/neodictabert**: NeoDictaBERT from Dicta (latest Hebrew BERT)
            
            **Note**: 
            - First run will download the selected model (~500MB)
            - Some models may not be in bert_score's registry; if you encounter errors, use **avichr/heBERT**
            - Models require `transformers` library: `pip install transformers`
            """)

st.markdown("---")

# Compute button
if st.button("🔬 Compute Metrics", type="primary", use_container_width=True):
    if not source_text or not source_text.strip():
        st.error("⚠️ Please provide a source document")
    elif not summary_text or not summary_text.strip():
        st.error("⚠️ Please provide a generated summary")
    else:
        with st.spinner("Computing metrics..."):
            results = {
                "source_length": len(tokenize(source_text)),
                "summary_length": len(tokenize(summary_text)),
                "metrics": {}
            }

            # Compute ROUGE
            if compute_rouge:
                results["metrics"]["ROUGE"] = {}

                if "ROUGE-1" in rouge_variants:
                    results["metrics"]["ROUGE"]["ROUGE-1"] = compute_rouge_n(source_text, summary_text, 1)

                if "ROUGE-2" in rouge_variants:
                    results["metrics"]["ROUGE"]["ROUGE-2"] = compute_rouge_n(source_text, summary_text, 2)

                if "ROUGE-L" in rouge_variants:
                    results["metrics"]["ROUGE"]["ROUGE-L"] = compute_rouge_l(source_text, summary_text)

            # Compute BLEU
            if compute_bleu_check:
                results["metrics"]["BLEU"] = compute_bleu(source_text, summary_text, bleu_max_n)

            # Compute ChrF
            if compute_chrf_check:
                results["metrics"]["ChrF"] = compute_chrf(source_text, summary_text, chrf_max_n)

            # Compute BERTScore
            if compute_bertscore_check:
                with st.spinner(f"Loading {bert_model} and computing BERTScore..."):
                    results["metrics"]["BERTScore"] = compute_bertscore(source_text, summary_text, bert_model)

        # Display results
        st.success("✅ Metrics computed successfully!")

        st.markdown("---")
        st.subheader("📈 Results")

        # Basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Source Length", f"{results['source_length']} words")
        with col2:
            st.metric("Summary Length", f"{results['summary_length']} words")
        with col3:
            compression_ratio = (results['summary_length'] / results['source_length'] * 100) if results['source_length'] > 0 else 0
            st.metric("Compression Ratio", f"{compression_ratio:.1f}%")

        st.markdown("---")

        # Display metrics
        if "ROUGE" in results["metrics"]:
            st.subheader("🔴 ROUGE Scores")
            st.markdown("""
                **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**  
                Measures n-gram overlap between summary and source. Higher recall indicates better content coverage.
                """)

            for variant, scores in results["metrics"]["ROUGE"].items():
                with st.expander(f"📊 {variant}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision", f"{scores['precision']:.4f}")
                    with col2:
                        st.metric("Recall", f"{scores['recall']:.4f}")
                    with col3:
                        st.metric("F1-Score", f"{scores['f1']:.4f}")

                    # Visual bar for F1
                    st.progress(scores['f1'], text=f"F1: {scores['f1']:.2%}")

        if "BLEU" in results["metrics"]:
            st.markdown("---")
            st.subheader("🔵 BLEU Score")
            st.markdown("""
            **BLEU (Bilingual Evaluation Understudy)**  
            Measures n-gram precision with brevity penalty. Originally for machine translation, penalizes overly short outputs.
            """)

            bleu_data = results["metrics"]["BLEU"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall BLEU", f"{bleu_data['bleu']:.4f}", help="Geometric mean of n-gram precisions × brevity penalty")
            with col2:
                st.metric("Brevity Penalty", f"{bleu_data['brevity_penalty']:.4f}", help="Penalty for summaries shorter than reference")

            st.progress(bleu_data['bleu'], text=f"BLEU: {bleu_data['bleu']:.2%}")

            with st.expander("📊 Per-N-gram Scores"):
                cols = st.columns(len(bleu_data['scores']))
                for idx, (ngram, score) in enumerate(bleu_data['scores'].items()):
                    with cols[idx]:
                        st.metric(ngram.upper(), f"{score:.4f}")

        if "ChrF" in results["metrics"]:
            st.markdown("---")
            st.subheader("🟢 ChrF Score")
            st.markdown("""
            **ChrF / ChrF++ (Character N-gram F-score)**  
            Character-level n-gram matching. Especially useful for morphologically rich languages like Hebrew.
            """)

            chrf_data = results["metrics"]["ChrF"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ChrF Score", f"{chrf_data['chrf']:.2f}", help="Character F-score (0-100)")
            with col2:
                st.metric("Char Precision", f"{chrf_data['precision']:.4f}")
            with col3:
                st.metric("Char Recall", f"{chrf_data['recall']:.4f}")

            st.progress(chrf_data['chrf'] / 100, text=f"ChrF: {chrf_data['chrf']:.1f}%")

        if "BERTScore" in results["metrics"]:
            st.markdown("---")
            st.subheader("🟣 BERTScore")
            st.markdown("""
            **BERTScore (Contextual Embedding Similarity)**  
            Measures semantic similarity using contextual embeddings from Hebrew BERT.  
            Unlike n-gram metrics, BERTScore captures paraphrases and semantic equivalence.
            """)

            bertscore_data = results["metrics"]["BERTScore"]

            if "error" in bertscore_data and bertscore_data["error"]:
                st.error(f"⚠️ BERTScore computation failed: {bertscore_data['error']}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision", f"{bertscore_data['precision']:.4f}", help="Avg max similarity: candidate → reference tokens")
                with col2:
                    st.metric("Recall", f"{bertscore_data['recall']:.4f}", help="Avg max similarity: reference → candidate tokens")
                with col3:
                    st.metric("F1-Score", f"{bertscore_data['f1']:.4f}", help="Harmonic mean of precision and recall")

                st.progress(bertscore_data['f1'], text=f"BERTScore F1: {bertscore_data['f1']:.2%}")

                # Interpretation guide
                with st.expander("📊 Score Interpretation"):
                    st.markdown("""
                    **BERTScore ranges:**
                    - **> 0.85**: Very high semantic similarity (near-perfect paraphrase)
                    - **0.75 - 0.85**: Good semantic overlap (acceptable summary)
                    - **0.60 - 0.75**: Moderate similarity (some semantic drift)
                    - **< 0.60**: Low similarity (significant content mismatch)
                    
                    **Note**: BERTScore measures semantic similarity, not factual correctness.  
                    A summary can have high BERTScore but still contain factual errors.
                    """)

        st.markdown("---")

        # Download results
        st.subheader("📥 Export Results")

        export_data = {
            "source_text": source_text,
            "summary_text": summary_text,
            "statistics": {
                "source_length_words": results["source_length"],
                "summary_length_words": results["summary_length"],
                "compression_ratio": round(compression_ratio, 2)
            },
            "metrics": results["metrics"]
        }

        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
            "📥 Download Results (JSON)",
            json_str.encode('utf-8'),
            "evaluation_results.json",
            "application/json",
            help="Download all metrics and texts as JSON"
        )

        with col2:
            # Create simple text report
            report_lines = [
        "=== Correctness Evaluation Report ===",
        f"\nSource Length: {results['source_length']} words",
        f"Summary Length: {results['summary_length']} words",
        f"Compression: {compression_ratio:.1f}%",
        "\n--- Metrics ---"
    ]

            if "ROUGE" in results["metrics"]:
                report_lines.append("\nROUGE Scores:")
                for variant, scores in results["metrics"]["ROUGE"].items():
                    report_lines.append(f"  {variant}: P={scores['precision']:.4f}, R={scores['recall']:.4f}, F1={scores['f1']:.4f}")

            if "BLEU" in results["metrics"]:
                report_lines.append(f"\nBLEU Score: {results['metrics']['BLEU']['bleu']:.4f}")
                report_lines.append(f"  Brevity Penalty: {results['metrics']['BLEU']['brevity_penalty']:.4f}")

            if "ChrF" in results["metrics"]:
                report_lines.append(f"\nChrF Score: {results['metrics']['ChrF']['chrf']:.2f}")

            if "BERTScore" in results["metrics"] and "error" not in results["metrics"]["BERTScore"]:
                bs = results["metrics"]["BERTScore"]
                report_lines.append(f"\nBERTScore: P={bs['precision']:.4f}, R={bs['recall']:.4f}, F1={bs['f1']:.4f}")

            report_text = "\n".join(report_lines)

            st.download_button(
        "📄 Download Report (TXT)",
        report_text.encode('utf-8'),
        "evaluation_report.txt",
        "text/plain",
        help="Download simple text report"
    )

            # else:
            #     st.info("👆 Enter source and summary texts above, select metrics, then click 'Compute Metrics'")

# ===== BATCH EVALUATION TAB =====
with tab_batch:
    st.markdown("---")
    st.subheader("📊 Batch Evaluation from CSV")
    st.markdown("Upload a CSV file with multiple samples to evaluate metrics in batch.")
    
    # File upload
    batch_file = st.file_uploader(
        "Upload CSV file with premise, hypothesis, and label columns",
        type=["csv"],
        key="batch_csv",
        help="CSV should contain: premise, hypothesis, label (and optionally other columns)"
    )
    
    if batch_file is not None:
        import pandas as pd
        
        try:
            df_batch = pd.read_csv(batch_file)
            st.success(f"✅ Loaded {len(df_batch)} records")
            
            # Validate required columns
            required_cols = ['premise', 'hypothesis', 'label']
            missing_cols = [col for col in required_cols if col not in df_batch.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                # Show preview
                with st.expander("🔍 Dataset Preview (first 5 rows)"):
                    st.dataframe(df_batch.head(), use_container_width=True)
                
                # Configuration
                st.markdown("---")
                st.subheader("⚙️ Evaluation Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sample size selection
                    max_samples = len(df_batch)
                    sample_size = st.slider(
                        "Number of samples to evaluate:",
                        min_value=1,
                        max_value=max_samples,
                        value=min(100, max_samples),
                        key="batch_sample_size",
                        help="Select how many samples from the dataset to evaluate"
                    )
                    
                    st.info(f"📊 Will evaluate {sample_size} out of {max_samples} samples")
                
                with col2:
                    # Metric selection
                    st.markdown("**Select Metrics:**")
                    
                    batch_rouge = st.checkbox("ROUGE-1", value=True, key="batch_rouge1")
                    batch_rouge2 = st.checkbox("ROUGE-2", value=True, key="batch_rouge2")
                    batch_rougel = st.checkbox("ROUGE-L", value=True, key="batch_rougel")
                    batch_bleu = st.checkbox("BLEU", value=True, key="batch_bleu")
                    batch_chrf = st.checkbox("ChrF", value=True, key="batch_chrf")
                    
                    # BERTScore option (expensive)
                    if BERTSCORE_AVAILABLE:
                        batch_bertscore = st.checkbox(
                            "BERTScore (⚠️ slow for large batches)", 
                            value=False, 
                            key="batch_bertscore",
                            help="BERTScore requires loading BERT model and is computationally expensive"
                        )
                        
                        if batch_bertscore:
                            batch_bert_model = st.selectbox(
                                "Hebrew BERT Model:",
                                ["avichr/heBERT", "onlplab/alephbert-base", "dicta-il/neodictabert"],
                                index=0,
                                key="batch_bert_model"
                            )
                    else:
                        batch_bertscore = False
                
                # Compute button
                if st.button("🚀 Evaluate Batch", type="primary", use_container_width=True):
                    # Select samples
                    df_eval = df_batch.head(sample_size).copy()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results_list = []
                    
                    for idx, row in df_eval.iterrows():
                        status_text.text(f"Evaluating sample {idx + 1}/{sample_size}...")
                        progress_bar.progress((idx + 1) / sample_size)
                        
                        premise = str(row['premise'])
                        hypothesis = str(row['hypothesis'])
                        
                        result = {'index': idx}
                        
                        # Compute ROUGE-1
                        if batch_rouge:
                            rouge1 = compute_rouge_n(premise, hypothesis, 1)
                            result['rouge1_precision'] = rouge1['precision']
                            result['rouge1_recall'] = rouge1['recall']
                            result['rouge1_f1'] = rouge1['f1']
                        
                        # Compute ROUGE-2
                        if batch_rouge2:
                            rouge2 = compute_rouge_n(premise, hypothesis, 2)
                            result['rouge2_precision'] = rouge2['precision']
                            result['rouge2_recall'] = rouge2['recall']
                            result['rouge2_f1'] = rouge2['f1']
                        
                        # Compute ROUGE-L
                        if batch_rougel:
                            rougel = compute_rouge_l(premise, hypothesis)
                            result['rougel_precision'] = rougel['precision']
                            result['rougel_recall'] = rougel['recall']
                            result['rougel_f1'] = rougel['f1']
                        
                        # Compute BLEU
                        if batch_bleu:
                            bleu_result = compute_bleu(premise, hypothesis, max_n=4)
                            result['bleu'] = bleu_result['bleu']
                            result['bleu_bp'] = bleu_result['brevity_penalty']
                        
                        # Compute ChrF
                        if batch_chrf:
                            chrf_result = compute_chrf(premise, hypothesis, n=6)
                            result['chrf'] = chrf_result['chrf']
                        
                        # Compute BERTScore (expensive)
                        if batch_bertscore:
                            bert_result = compute_bertscore(premise, hypothesis, batch_bert_model)
                            if 'error' not in bert_result:
                                result['bertscore_precision'] = bert_result['precision']
                                result['bertscore_recall'] = bert_result['recall']
                                result['bertscore_f1'] = bert_result['f1']
                        
                        results_list.append(result)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Convert results to DataFrame
                    results_df = pd.DataFrame(results_list)
                    
                    # Merge with original data
                    df_with_metrics = df_eval.copy()
                    
                    # Add metric columns
                    for col in results_df.columns:
                        if col != 'index':
                            df_with_metrics[col] = results_df[col].values
                    
                    st.success(f"✅ Evaluation complete! Processed {sample_size} samples.")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📈 Results Summary")
                    
                    # Calculate average scores
                    metric_cols = [col for col in results_df.columns if col != 'index']
                    
                    if metric_cols:
                        avg_scores = {}
                        for col in metric_cols:
                            avg_scores[col] = results_df[col].mean()
                        
                        # Display averages
                        st.markdown("**Average Scores Across All Samples:**")
                        
                        cols = st.columns(min(4, len(avg_scores)))
                        for idx, (metric_name, avg_value) in enumerate(avg_scores.items()):
                            with cols[idx % 4]:
                                st.metric(metric_name.upper(), f"{avg_value:.4f}")
                        
                        # Kendall Tau Correlation Analysis
                        st.markdown("---")
                        st.subheader("📊 Kendall Tau Correlation (Metric vs Label)")
                        st.markdown("""
                        **Kendall τ measures correlation between metric scores and correctness labels:**
                        - **τ ≈ 1.0**: Metric strongly predicts correctness (higher score → label=1)
                        - **τ ≈ 0.0**: No correlation (metric doesn't reflect correctness)
                        - **τ < 0.0**: Negative correlation (metric misleading)
                        """)
                        
                        # Convert labels to binary (0/1) if they are strings
                        labels = df_eval['label'].values
                        label_binary = []
                        
                        for label in labels:
                            if isinstance(label, str):
                                # Convert string labels to binary
                                if label.lower() in ['entailment', '1', 'true', 'correct']:
                                    label_binary.append(1)
                                elif label.lower() in ['contradiction', '0', 'false', 'incorrect']:
                                    label_binary.append(0)
                                else:
                                    label_binary.append(None)  # Unknown label
                            else:
                                label_binary.append(int(label))
                        
                        # Filter out None values
                        valid_indices = [i for i, lbl in enumerate(label_binary) if lbl is not None]
                        
                        if len(valid_indices) > 1:
                            label_binary_clean = [label_binary[i] for i in valid_indices]
                            
                            # Calculate Kendall Tau for each metric
                            kendall_results = {}
                            
                            # BLEU vs Label
                            if 'bleu' in results_df.columns:
                                bleu_scores = [results_df.iloc[i]['bleu'] for i in valid_indices]
                                tau, p_value = kendalltau(bleu_scores, label_binary_clean)
                                kendall_results['BLEU'] = {'tau': tau, 'p_value': p_value}
                            
                            # ROUGE-1 F1 vs Label
                            if 'rouge1_f1' in results_df.columns:
                                rouge1_scores = [results_df.iloc[i]['rouge1_f1'] for i in valid_indices]
                                tau, p_value = kendalltau(rouge1_scores, label_binary_clean)
                                kendall_results['ROUGE-1 F1'] = {'tau': tau, 'p_value': p_value}
                            
                            # ROUGE-2 F1 vs Label
                            if 'rouge2_f1' in results_df.columns:
                                rouge2_scores = [results_df.iloc[i]['rouge2_f1'] for i in valid_indices]
                                tau, p_value = kendalltau(rouge2_scores, label_binary_clean)
                                kendall_results['ROUGE-2 F1'] = {'tau': tau, 'p_value': p_value}
                            
                            # ROUGE-L F1 vs Label
                            if 'rougel_f1' in results_df.columns:
                                rougel_scores = [results_df.iloc[i]['rougel_f1'] for i in valid_indices]
                                tau, p_value = kendalltau(rougel_scores, label_binary_clean)
                                kendall_results['ROUGE-L F1'] = {'tau': tau, 'p_value': p_value}
                            
                            # ChrF vs Label
                            if 'chrf' in results_df.columns:
                                chrf_scores = [results_df.iloc[i]['chrf'] for i in valid_indices]
                                tau, p_value = kendalltau(chrf_scores, label_binary_clean)
                                kendall_results['ChrF'] = {'tau': tau, 'p_value': p_value}
                            
                            # BERTScore F1 vs Label
                            if 'bertscore_f1' in results_df.columns:
                                bert_scores = [results_df.iloc[i]['bertscore_f1'] for i in valid_indices]
                                tau, p_value = kendalltau(bert_scores, label_binary_clean)
                                kendall_results['BERTScore F1'] = {'tau': tau, 'p_value': p_value}
                            
                            # Display Kendall Tau results
                            if kendall_results:
                                cols = st.columns(min(3, len(kendall_results)))
                                for idx, (metric_name, result) in enumerate(kendall_results.items()):
                                    with cols[idx % 3]:
                                        tau_value = result['tau']
                                        p_value = result['p_value']
                                        
                                        # Color-code based on tau value
                                        if tau_value > 0.5:
                                            delta_color = "normal"  # Green
                                            interpretation = "Strong positive"
                                        elif tau_value > 0.3:
                                            delta_color = "normal"
                                            interpretation = "Moderate positive"
                                        elif tau_value > 0.1:
                                            delta_color = "off"
                                            interpretation = "Weak positive"
                                        elif tau_value > -0.1:
                                            delta_color = "off"
                                            interpretation = "No correlation"
                                        else:
                                            delta_color = "inverse"  # Red
                                            interpretation = "Negative"
                                        
                                        st.metric(
                                            f"{metric_name}",
                                            f"τ = {tau_value:.4f}",
                                            f"p = {p_value:.4f}",
                                            delta_color=delta_color,
                                            help=f"{interpretation} correlation"
                                        )
                                
                                # Interpretation guide
                                with st.expander("📖 Understanding Kendall Tau Results"):
                                    st.markdown("""
                                    **Kendall τ (Tau) Interpretation:**
                                    
                                    **Positive τ (Good):**
                                    - **τ > 0.7**: Excellent - Metric strongly predicts correctness
                                    - **τ > 0.5**: Strong - Metric reliably reflects faithfulness
                                    - **τ > 0.3**: Moderate - Some correlation with correctness
                                    - **τ > 0.1**: Weak - Limited correlation
                                    
                                    **No Correlation:**
                                    - **-0.1 < τ < 0.1**: Metric doesn't reflect correctness
                                    
                                    **Negative τ (Bad):**
                                    - **τ < -0.1**: Metric is misleading (high scores for incorrect summaries)
                                    
                                    **P-value:**
                                    - **p < 0.05**: Statistically significant (reliable result)
                                    - **p > 0.05**: Not significant (result may be random)
                                    
                                    **What This Means:**
                                    - High τ means the metric can be used as a proxy for factual correctness
                                    - Low τ means the metric measures surface similarity, not factuality
                                    - Negative τ means the metric is actively misleading
                                    
                                    **Example:**
                                    - BLEU τ = 0.8 → High BLEU strongly indicates correct summary
                                    - ROUGE τ = 0.2 → ROUGE weakly correlates with correctness
                                    - BERTScore τ = 0.9 → BERTScore excellent for faithfulness detection
                                    """)
                            else:
                                st.info("ℹ️ No metrics available for correlation analysis")
                        else:
                            st.warning("⚠️ Need at least 2 valid samples with binary labels (0/1 or entailment/contradiction) for correlation analysis")
                        
                        # Show detailed results
                        st.markdown("---")
                        st.subheader("📋 Detailed Results")
                        
                        # Show results table
                        display_cols = ['premise', 'hypothesis', 'label'] + metric_cols
                        available_cols = [col for col in display_cols if col in df_with_metrics.columns]
                        
                        with st.expander("🔍 View All Results", expanded=True):
                            st.dataframe(df_with_metrics[available_cols], use_container_width=True)
                        
                        # Download options
                        st.markdown("---")
                        st.subheader("📥 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download full CSV with metrics
                            csv_output = df_with_metrics.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Full Results (CSV)",
                                csv_output,
                                "batch_evaluation_results.csv",
                                "text/csv",
                                help="Download CSV with all original columns plus computed metrics"
                            )
                        
                        with col2:
                            # Download summary statistics
                            summary_data = {
                                'metric': list(avg_scores.keys()),
                                'average': list(avg_scores.values()),
                                'min': [results_df[col].min() for col in avg_scores.keys()],
                                'max': [results_df[col].max() for col in avg_scores.keys()],
                                'std': [results_df[col].std() for col in avg_scores.keys()]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
                            
                            st.download_button(
                                "📊 Download Summary Statistics (CSV)",
                                summary_csv,
                                "batch_evaluation_summary.csv",
                                "text/csv",
                                help="Download summary statistics for each metric"
                            )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("📁 Upload a CSV file to start batch evaluation")
        st.markdown("""
        **Expected CSV Format:**
        - **Required columns**: premise, hypothesis, label
        - **Optional columns**: Any additional columns will be preserved in the output
        
        **Example:**
        ```
        premise,hypothesis,label
        "Source text 1","Summary 1","entailment"
        "Source text 2","Summary 2","contradiction"
        ```
        
        **Features:**
        - 📊 Select how many samples to evaluate (1 to all)
        - ☑️ Choose which metrics to compute
        - 📥 Download results with metrics as new columns
        - 📈 View summary statistics (average, min, max, std)
        - ⚡ Fast evaluation for surface-level metrics
        - 🐌 BERTScore available (slower, optional)
        """)

# Help section
with st.expander("ℹ️ Metric Explanations"):
    st.markdown("""
        ### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
        - **ROUGE-1**: Unigram (single word) overlap
        - **ROUGE-2**: Bigram (two-word sequence) overlap
        - **ROUGE-L**: Longest Common Subsequence matching
        - **Focus**: Recall (how much of the source is covered)
        - **Range**: 0.0 to 1.0 (higher is better)
        
        ### BLEU (Bilingual Evaluation Understudy)
        - Measures n-gram precision (1-gram through 4-gram)
        - Applies brevity penalty for short outputs
        - Originally designed for machine translation
        - **Focus**: Precision (accuracy of generated text)
        - **Range**: 0.0 to 1.0 (higher is better)
        
        ### ChrF / ChrF++
        - Character-level n-gram F-score
        - Robust for morphologically rich languages (Hebrew, Arabic, etc.)
        - Less sensitive to tokenization errors
        - **Focus**: Character-level similarity with beta=2 (favors recall)
        - **Range**: 0 to 100 (higher is better)
        
        ### BERTScore
        - Measures semantic similarity using contextual embeddings from BERT
        - **How it works**:
          1. Encode both texts with Hebrew BERT → contextual embeddings
          2. Compute pairwise cosine similarity between all token pairs
          3. Greedy matching: each token finds its best match
          4. Average similarities → Precision, Recall, F1
        - **Focus**: Semantic equivalence (captures paraphrases)
        - **Range**: 0.0 to 1.0 (higher is better)
        - **Note**: Requires downloading Hebrew BERT model (~500MB first time)
        
        ### When to Use Each Metric
        - **ROUGE**: Standard for abstractive summarization evaluation
        - **BLEU**: When precision matters (avoiding hallucinations)
        - **ChrF**: For Hebrew text or when dealing with morphological variations
        - **BERTScore**: When evaluating semantic similarity beyond exact word matches
        """)

with st.expander("💡 Tips for Interpretation"):
    st.markdown("""
        ### Surface-Level Metrics
        - **ROUGE-1 > 0.4**: Good content coverage
        - **ROUGE-2 > 0.2**: Good phrase-level matching
        - **BLEU > 0.3**: Reasonable precision
        - **ChrF > 40**: Good character-level similarity
        
        ### Semantic Metrics
        - **BERTScore F1 > 0.85**: Excellent semantic similarity
        - **BERTScore F1 > 0.75**: Good semantic overlap
        - **BERTScore F1 < 0.60**: Significant semantic drift
        
        **Note**: These are rough guidelines. Actual "good" scores depend on:
        - Language characteristics
        - Summary type (extractive vs abstractive)
        - Domain and task requirements
        - Source document length and complexity
        
        **Important**: BERTScore measures semantic similarity, not factual accuracy.  
        Always combine with human evaluation for factual consistency.
        """)

# ===== CLASSIFIER PREDICTIONS TAB =====
with tab_classifier:
    st.markdown("---")
    st.subheader("🎯 Classifier Prediction Analysis")
    st.markdown("Analyze correlation between model confidence scores and true labels using Kendall Tau.")
    
    # File upload
    classifier_file = st.file_uploader(
        "Upload CSV with classifier predictions",
        type=["csv"],
        key="classifier_csv",
        help="CSV should contain: premise, hypothesis, true_label, predicted_label, prob_contradiction, prob_entailment"
    )
    
    if classifier_file is not None:
        import pandas as pd
        import numpy as np
        
        try:
            df_classifier = pd.read_csv(classifier_file)
            st.success(f"✅ Loaded {len(df_classifier)} predictions")
            
            # Validate required columns
            required_cols = ['premise', 'hypothesis', 'true_label', 'predicted_label', 'prob_contradiction', 'prob_entailment']
            missing_cols = [col for col in required_cols if col not in df_classifier.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info(f"Available columns: {list(df_classifier.columns)}")
            else:
                # Show preview
                with st.expander("🔍 Dataset Preview (first 5 rows)"):
                    st.dataframe(df_classifier.head(), use_container_width=True)
                
                # Basic statistics
                st.markdown("---")
                st.subheader("📊 Dataset Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Samples", len(df_classifier))
                
                with col2:
                    # Count correct predictions
                    correct = (df_classifier['true_label'] == df_classifier['predicted_label']).sum()
                    accuracy = correct / len(df_classifier) * 100
                    st.metric("Accuracy", f"{accuracy:.2f}%")
                
                with col3:
                    # Label distribution
                    true_dist = df_classifier['true_label'].value_counts()
                    st.metric("True Label Distribution", f"{true_dist.to_dict()}")
                
                with col4:
                    # Average confidence
                    max_probs = df_classifier[['prob_contradiction', 'prob_entailment']].max(axis=1)
                    avg_confidence = max_probs.mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.4f}")
                
                # Kendall Tau Analysis
                st.markdown("---")
                st.subheader("📈 Kendall Tau Correlation Analysis")
                st.markdown("""
                **Measures correlation between model confidence and label correctness:**
                - High τ → Model is well-calibrated (confident when correct, uncertain when wrong)
                - Low τ → Model confidence doesn't reflect accuracy
                """)
                
                # Convert labels to binary
                labels_true = df_classifier['true_label'].values
                labels_pred = df_classifier['predicted_label'].values
                
                # Create binary correctness indicator
                is_correct = (labels_true == labels_pred).astype(int)
                
                # Get probabilities
                prob_contradiction = df_classifier['prob_contradiction'].values
                prob_entailment = df_classifier['prob_entailment'].values
                
                # Calculate max probability (model confidence)
                confidence_scores = np.maximum(prob_contradiction, prob_entailment)
                
                # Calculate Kendall Tau: Confidence vs Correctness
                tau_confidence, p_confidence = kendalltau(confidence_scores, is_correct)
                
                # Calculate Kendall Tau for each probability column vs binary labels
                # Convert string labels to binary if needed
                label_binary = []
                for label in labels_true:
                    if isinstance(label, str):
                        if label.lower() in ['entailment', '1', 'true']:
                            label_binary.append(1)
                        elif label.lower() in ['contradiction', '0', 'false']:
                            label_binary.append(0)
                        else:
                            label_binary.append(None)
                    else:
                        label_binary.append(int(label))
                
                # Filter valid labels
                valid_indices = [i for i, lbl in enumerate(label_binary) if lbl is not None]
                
                if len(valid_indices) > 1:
                    label_binary_clean = [label_binary[i] for i in valid_indices]
                    prob_entailment_clean = [prob_entailment[i] for i in valid_indices]
                    prob_contradiction_clean = [prob_contradiction[i] for i in valid_indices]
                    
                    # Kendall Tau: prob_entailment vs true_label
                    tau_entailment, p_entailment = kendalltau(prob_entailment_clean, label_binary_clean)
                    
                    # Kendall Tau: prob_contradiction vs true_label (inverse)
                    tau_contradiction, p_contradiction = kendalltau(prob_contradiction_clean, [1 - lbl for lbl in label_binary_clean])
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Confidence vs Correctness**")
                        interpretation_conf = ""
                        if tau_confidence > 0.5:
                            interpretation_conf = "Strong - Well calibrated ✅"
                            delta_color = "normal"
                        elif tau_confidence > 0.3:
                            interpretation_conf = "Moderate calibration"
                            delta_color = "normal"
                        elif tau_confidence > 0.1:
                            interpretation_conf = "Weak calibration"
                            delta_color = "off"
                        else:
                            interpretation_conf = "Poor calibration ⚠️"
                            delta_color = "inverse"
                        
                        st.metric(
                            "Kendall τ",
                            f"{tau_confidence:.4f}",
                            f"p = {p_confidence:.4f}",
                            delta_color=delta_color
                        )
                        st.caption(interpretation_conf)
                    
                    with col2:
                        st.markdown("**prob_entailment vs Label**")
                        interpretation_ent = ""
                        if tau_entailment > 0.5:
                            interpretation_ent = "Strong correlation ✅"
                            delta_color = "normal"
                        elif tau_entailment > 0.3:
                            interpretation_ent = "Moderate correlation"
                            delta_color = "normal"
                        elif tau_entailment > 0.1:
                            interpretation_ent = "Weak correlation"
                            delta_color = "off"
                        elif tau_entailment > -0.1:
                            interpretation_ent = "No correlation"
                            delta_color = "off"
                        else:
                            interpretation_ent = "Negative correlation ⚠️"
                            delta_color = "inverse"
                        
                        st.metric(
                            "Kendall τ",
                            f"{tau_entailment:.4f}",
                            f"p = {p_entailment:.4f}",
                            delta_color=delta_color
                        )
                        st.caption(interpretation_ent)
                    
                    with col3:
                        st.markdown("**prob_contradiction vs Label**")
                        interpretation_con = ""
                        if tau_contradiction > 0.5:
                            interpretation_con = "Strong correlation ✅"
                            delta_color = "normal"
                        elif tau_contradiction > 0.3:
                            interpretation_con = "Moderate correlation"
                            delta_color = "normal"
                        elif tau_contradiction > 0.1:
                            interpretation_con = "Weak correlation"
                            delta_color = "off"
                        elif tau_contradiction > -0.1:
                            interpretation_con = "No correlation"
                            delta_color = "off"
                        else:
                            interpretation_con = "Negative correlation ⚠️"
                            delta_color = "inverse"
                        
                        st.metric(
                            "Kendall τ",
                            f"{tau_contradiction:.4f}",
                            f"p = {p_contradiction:.4f}",
                            delta_color=delta_color
                        )
                        st.caption(interpretation_con)
                    
                    # Detailed analysis
                    st.markdown("---")
                    st.subheader("🔍 Detailed Analysis")
                    
                    # Confidence by correctness
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Confidence Scores by Correctness**")
                        
                        correct_mask = is_correct == 1
                        wrong_mask = is_correct == 0
                        
                        avg_conf_correct = confidence_scores[correct_mask].mean() if correct_mask.sum() > 0 else 0
                        avg_conf_wrong = confidence_scores[wrong_mask].mean() if wrong_mask.sum() > 0 else 0
                        
                        conf_data = pd.DataFrame({
                            'Category': ['Correct Predictions', 'Wrong Predictions'],
                            'Avg Confidence': [avg_conf_correct, avg_conf_wrong],
                            'Count': [correct_mask.sum(), wrong_mask.sum()]
                        })
                        
                        st.dataframe(conf_data, use_container_width=True)
                        
                        # Interpretation
                        if avg_conf_correct > avg_conf_wrong + 0.1:
                            st.success("✅ Model is well-calibrated: Higher confidence on correct predictions")
                        elif avg_conf_correct > avg_conf_wrong:
                            st.info("ℹ️ Model slightly more confident on correct predictions")
                        else:
                            st.warning("⚠️ Model is poorly calibrated: Similar or higher confidence on wrong predictions")
                    
                    with col2:
                        st.markdown("**Probability Distribution by Label**")
                        
                        # Group by true label
                        label_stats = []
                        for label in df_classifier['true_label'].unique():
                            mask = df_classifier['true_label'] == label
                            label_stats.append({
                                'True Label': label,
                                'Avg prob_entailment': prob_entailment[mask].mean(),
                                'Avg prob_contradiction': prob_contradiction[mask].mean(),
                                'Count': mask.sum()
                            })
                        
                        label_stats_df = pd.DataFrame(label_stats)
                        st.dataframe(label_stats_df, use_container_width=True)
                    
                    # Interpretation guide
                    with st.expander("📖 Understanding the Results"):
                        st.markdown("""
                        **Confidence vs Correctness (Calibration):**
                        - **High τ (> 0.5)**: Model knows when it's right vs wrong ✅
                        - **Low τ (< 0.3)**: Model overconfident or underconfident ⚠️
                        - **Ideal**: High confidence on correct, low confidence on errors
                        
                        **prob_entailment vs Label:**
                        - **High τ**: prob_entailment reliably indicates label=entailment
                        - **Low τ**: Probability scores don't reflect true labels well
                        
                        **prob_contradiction vs Label:**
                        - **High τ**: prob_contradiction reliably indicates label=contradiction
                        - **Low τ**: Probability scores unreliable
                        
                        **What Good Results Look Like:**
                        ```
                        Confidence vs Correctness: τ > 0.5 (well-calibrated)
                        prob_entailment vs Label: τ > 0.7 (strong correlation)
                        prob_contradiction vs Label: τ > 0.7 (strong correlation)
                        
                        Avg confidence (correct): > 0.80
                        Avg confidence (wrong): < 0.60
                        ```
                        
                        **If Results Are Poor:**
                        - Consider temperature calibration
                        - May need more training data
                        - Check for class imbalance
                        - Review model architecture
                        """)
                    
                    # Download results
                    st.markdown("---")
                    st.subheader("📥 Download Analysis")
                    
                    # Create analysis summary
                    analysis_summary = {
                        'total_samples': len(df_classifier),
                        'accuracy': float(accuracy / 100),
                        'avg_confidence': float(avg_confidence),
                        'kendall_tau_confidence_correctness': {
                            'tau': float(tau_confidence),
                            'p_value': float(p_confidence)
                        },
                        'kendall_tau_prob_entailment_label': {
                            'tau': float(tau_entailment),
                            'p_value': float(p_entailment)
                        },
                        'kendall_tau_prob_contradiction_label': {
                            'tau': float(tau_contradiction),
                            'p_value': float(p_contradiction)
                        },
                        'avg_confidence_correct': float(avg_conf_correct),
                        'avg_confidence_wrong': float(avg_conf_wrong)
                    }
                    
                    analysis_json = json.dumps(analysis_summary, indent=2).encode('utf-8')
                    
                    st.download_button(
                        "📊 Download Analysis Summary (JSON)",
                        analysis_json,
                        "classifier_analysis.json",
                        "application/json",
                        help="Download correlation analysis results"
                    )
                
                else:
                    st.warning("⚠️ Need at least 2 valid samples with binary labels for correlation analysis")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.info("📁 Upload a CSV file with classifier predictions to start analysis")
        st.markdown("""
        **Expected CSV Format:**
        ```
        premise,hypothesis,true_label,predicted_label,prob_contradiction,prob_entailment
        "text1","claim1","entailment","entailment",0.1234,0.8766
        "text2","claim2","contradiction","contradiction",0.9234,0.0766
        ```
        
        **Required Columns:**
        - `premise`: Source text / article
        - `hypothesis`: Generated text / claim
        - `true_label`: Ground truth label
        - `predicted_label`: Model's predicted label
        - `prob_contradiction`: Probability score for contradiction
        - `prob_entailment`: Probability score for entailment
        
        **Analysis Provided:**
        - 📊 **Confidence vs Correctness**: Does the model know when it's right?
        - 🎯 **Probability Calibration**: Do probability scores reflect true labels?
        - 📈 **Kendall Tau Correlation**: Statistical measure of model calibration
        - 📋 **Detailed Statistics**: Confidence by correctness, probability distributions
        - 📥 **Export**: Download analysis summary as JSON
        
        **Use Cases:**
        - Model calibration analysis
        - Confidence threshold optimization
        - Model comparison (which is better calibrated?)
        - Detecting overconfident/underconfident predictions
        """)
