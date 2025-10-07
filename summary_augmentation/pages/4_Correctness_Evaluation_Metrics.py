import sys
import os
import json
import streamlit as st
from typing import Dict, List, Any, Optional
from collections import Counter
import re

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

else:
    st.info("👆 Enter source and summary texts above, select metrics, then click 'Compute Metrics'")

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
