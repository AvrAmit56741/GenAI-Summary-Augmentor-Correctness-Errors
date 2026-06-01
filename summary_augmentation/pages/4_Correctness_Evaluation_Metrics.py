import sys
import os
import json
import math
import time
import queue
import threading
import zipfile
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter
import re
from scipy.stats import kendalltau
from dotenv import load_dotenv

# Load .env from project root (two levels above this file)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"), override=True)

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (
    RougeCalculator, BleuCalculator, ChrfCalculator, TokenF1Calculator, NovelNgramCalculator,
    BertScoreCalculator, BERTSCORE_AVAILABLE,
    KendallTauCalculator, PointBiserialCalculator, ComparisonResult,
    ALL_METRICS_EXPLANATION, KENDALL_TAU_EXPLANATION,
)

st.set_page_config(page_title="Correctness Evaluation Metrics", layout="wide", page_icon="📏")

st.title("📏 Correctness Evaluation Metrics")
st.markdown(
    '<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>',
    unsafe_allow_html=True)

st.markdown("""
Evaluate summary quality using standard text-overlap metrics. Compare a generated summary against its source document.
""")

# Create tabs for single vs batch evaluation
tab_single, tab_ngram, tab_semantic, tab_classifier, tab_hallucination, tab_compare, tab_qafacteval, tab_summac, tab_geval, tab_vanilla, tab_bartscore, tab_mfact = st.tabs(["Single Evaluation", "N-gram", "Semantic", "Classifier Predictions", "Hallucination Metrics", "Compare Metrics", "QAFactEval", "SummaC", "G-EVAL", "LLM-as-Judge", "BARTScore", "MFact"])


# ── Shared helpers ────────────────────────────────────────────────────────────

def label_to_binary(label_series) -> list:
    """Convert entailment/contradiction labels to 1/0. Returns None for unknown values."""
    result = []
    for label_value in label_series:
        if isinstance(label_value, str):
            if label_value.lower() in ['entailment', '1', 'true', 'correct']:
                result.append(1)
            elif label_value.lower() in ['contradiction', '0', 'false', 'incorrect']:
                result.append(0)
            else:
                result.append(None)
        elif isinstance(label_value, float) and math.isnan(label_value):
            result.append(None)
        else:
            result.append(int(label_value))
    return result


def download_csv_json(
    csv_bytes: bytes,
    json_bytes: bytes,
    base_filename: str,
    csv_key: str = None,
    json_key: str = None,
):
    """Render side-by-side CSV and JSON download buttons."""
    col_csv, col_json = st.columns(2)
    with col_csv:
        st.download_button("📥 Download CSV", csv_bytes, f"{base_filename}.csv", "text/csv", key=csv_key)
    with col_json:
        st.download_button("📥 Download JSON", json_bytes, f"{base_filename}.json", "application/json", key=json_key)


def _build_run_config_md(metric_name: str, dataset_info: dict, config_rows: list) -> str:
    """Build a markdown run-configuration summary for download."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {metric_name} — Run Configuration",
        "",
        f"**Timestamp:** {ts}  ",
        f"**Dataset:** {dataset_info.get('name', 'unknown')}  ",
        f"**Dataset size:** {dataset_info.get('total_size', '?')} rows  ",
        f"**Samples evaluated:** {dataset_info.get('evaluated', '?')}  ",
        "",
        "---",
        "",
        "## Configuration",
        "",
    ]
    for label, value in config_rows:
        lines.append(f"- **{label}:** {value}")
    lines.append("")
    return "\n".join(lines)


def _build_llm_judge_summary_dataframe(
    result_log: list,
    sliced_dataframe: pd.DataFrame,
    core_field_names: dict,
) -> pd.DataFrame:
    """
    Build the results summary DataFrame for any LLM-as-Judge metric.
    Produces core scored columns (defined by core_field_names) followed by
    all metadata columns from the original CSV slice (everything except 'label',
    which is already included as a core field).
    """
    metadata_columns = [col for col in sliced_dataframe.columns if col != "label"]
    result_rows = []
    for row_offset, result_entry in enumerate(result_log):
        if result_entry is None:
            continue
        row_dict = {
            output_col: result_entry.get(source_key)
            for output_col, source_key in core_field_names.items()
        }
        original_row = sliced_dataframe.iloc[row_offset]
        for metadata_col in metadata_columns:
            row_dict[metadata_col] = original_row[metadata_col]
        result_rows.append(row_dict)
    return pd.DataFrame(result_rows)


def _run_llm_judge_batch(
    scorer_fn,
    serialize_fn,
    sliced_dataframe: pd.DataFrame,
    start_index: int,
    num_parallel_workers: int,
    stop_event: threading.Event,
    progress_label: str,
) -> tuple:
    """
    Shared execution loop for G-EVAL and Vanilla LLM-as-Judge.
    Submits all pairs to a ThreadPoolExecutor, updates a progress bar,
    and honours stop_event between completions.
    Returns (result_log, was_stopped).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dataclasses import asdict as dataclass_asdict

    total_samples = len(sliced_dataframe)
    result_log = [None] * total_samples
    completed_count = 0
    was_stopped = False

    progress_bar = st.progress(0, text=f"Starting {progress_label}...")

    def _score_single_row(row_offset, csv_row):
        if stop_event.is_set():
            return row_offset, None
        return row_offset, scorer_fn(
            source_document=str(csv_row['premise']),
            generated_summary=str(csv_row['hypothesis']),
        )

    with ThreadPoolExecutor(max_workers=num_parallel_workers) as pool:
        futures = {
            pool.submit(_score_single_row, i, row): i
            for i, (_, row) in enumerate(sliced_dataframe.iterrows())
        }
        for future in as_completed(futures):
            row_offset = futures[future]
            csv_row = sliced_dataframe.iloc[row_offset]
            try:
                _, eval_result = future.result()
                if eval_result is None:
                    result_log[row_offset] = {
                        "index": start_index + row_offset,
                        "premise": str(csv_row['premise']),
                        "hypothesis": str(csv_row['hypothesis']),
                        "label": str(csv_row.get('label', '')),
                        "score": None,
                        "error": "stopped",
                    }
                else:
                    result_log[row_offset] = dataclass_asdict(serialize_fn(
                        sample_index=start_index + row_offset,
                        source_document=str(csv_row['premise']),
                        generated_summary=str(csv_row['hypothesis']),
                        ground_truth_label=str(csv_row.get('label', '')),
                        eval_result=eval_result,
                    ))
            except Exception as sample_error:
                result_log[row_offset] = {
                    "index": start_index + row_offset,
                    "premise": str(csv_row['premise']),
                    "hypothesis": str(csv_row['hypothesis']),
                    "label": str(csv_row.get('label', '')),
                    "score": None,
                    "error": str(sample_error),
                }

            completed_count += 1
            if stop_event.is_set():
                was_stopped = True
                for pending_future in futures:
                    pending_future.cancel()
                break
            progress_bar.progress(
                completed_count / total_samples,
                text=f"{progress_label}: {completed_count} / {total_samples}",
            )

    progress_bar.empty()
    return result_log, was_stopped


def _batch_upload_and_range(key_prefix: str):
    """Shared: file upload + validation + row-range widgets. Returns (df, start, end) or (None, 0, 0)."""
    f = st.file_uploader("Upload CSV (premise, hypothesis, label columns required)", type=["csv"], key=f"{key_prefix}_csv")
    if f is None:
        st.info("📁 Upload a CSV file to start evaluation")
        return None, 0, 0
    try:
        df = pd.read_csv(f)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None, 0, 0
    missing = [c for c in ['premise', 'hypothesis', 'label'] if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        return None, 0, 0
    st.success(f"✅ Loaded {len(df)} records")
    with st.expander("🔍 Dataset Preview (first 5 rows)"):
        st.dataframe(df.head(), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        start = int(st.number_input("Start index:", 0, len(df) - 1, 0, 1, key=f"{key_prefix}_start"))
    with col2:
        end = int(st.number_input("End index:", start + 1, len(df), min(start + 100, len(df)), 1, key=f"{key_prefix}_end"))
    st.info(f"📊 Will evaluate rows {start}–{end - 1} ({end - start} samples)")
    return df, start, end


def _render_batch_results(results_df: pd.DataFrame, df_eval: pd.DataFrame, base_filename: str):
    """Shared: averages, Kendall Tau, detailed table, downloads."""
    metric_cols = [c for c in results_df.columns if c != 'index']
    avg_scores = {c: results_df[c].mean() for c in metric_cols}

    st.success(f"✅ Evaluation complete! Processed {len(df_eval)} samples.")
    st.markdown("---")
    st.subheader("📈 Results Summary")
    st.markdown("**Average Scores Across All Samples:**")
    if not avg_scores:
        st.warning("⚠️ No scores to display — all samples may have failed. Check the API connection and error messages above.")
        return
    cols = st.columns(min(4, len(avg_scores)))
    for idx, (name, val) in enumerate(avg_scores.items()):
        with cols[idx % 4]:
            st.metric(name.upper(), f"{val:.4f}")

    st.markdown("---")
    st.subheader("📊 Kendall Tau Correlation (Metric vs Label)")
    label_binary = label_to_binary(df_eval['label'].values)
    valid_indices = [i for i, lbl in enumerate(label_binary) if lbl is not None]
    if len(valid_indices) > 1:
        clean_labels = [label_binary[i] for i in valid_indices]
        candidate_cols = {
            'BLEU': 'bleu', 'ROUGE-1 F1': 'rouge1_f1', 'ROUGE-2 F1': 'rouge2_f1',
            'ROUGE-L F1': 'rougel_f1', 'ChrF': 'chrf', 'BERTScore F1': 'bertscore_f1',
            'Token F1': 'tokenf1_f1',
        }
        kendall_results = [
            KendallTauCalculator([results_df.iloc[i][col] for i in valid_indices], clean_labels, name).compute()
            for name, col in candidate_cols.items() if col in results_df.columns
        ]
        if kendall_results:
            cols = st.columns(min(3, len(kendall_results)))
            for idx, r in enumerate(kendall_results):
                dc = "normal" if r.statistic > 0.3 else ("inverse" if r.statistic < -0.1 else "off")
                with cols[idx % 3]:
                    st.metric(r.metric_name, f"τ = {r.statistic:.4f}", f"p = {r.p_value:.4f}", delta_color=dc, help=r.interpretation)
            with st.expander("📖 Understanding Kendall Tau Results"):
                st.markdown(KENDALL_TAU_EXPLANATION)
    else:
        st.warning("⚠️ Need at least 2 valid samples with binary labels for correlation analysis")

    df_out = df_eval.copy()
    for col in metric_cols:
        df_out[col] = results_df[col].values
    st.markdown("---")
    st.subheader("📋 Detailed Results")
    display_cols = [c for c in ['premise', 'hypothesis', 'label'] + metric_cols if c in df_out.columns]
    with st.expander("🔍 View All Results", expanded=True):
        st.dataframe(df_out[display_cols], use_container_width=True)

    st.markdown("---")
    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Full Results (CSV)", df_out.to_csv(index=False).encode(), f"{base_filename}.csv", "text/csv")
    with col2:
        summary_df = pd.DataFrame({
            'metric': list(avg_scores), 'average': list(avg_scores.values()),
            'min': [results_df[c].min() for c in avg_scores],
            'max': [results_df[c].max() for c in avg_scores],
            'std': [results_df[c].std() for c in avg_scores],
        })
        st.download_button("📊 Download Summary Statistics (CSV)", summary_df.to_csv(index=False).encode(), f"{base_filename}_summary.csv", "text/csv")


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
                    "source_length": len(source_text.strip().split()),
                    "summary_length": len(summary_text.strip().split()),
                    "metrics": {}
                }

                # Compute ROUGE
                if compute_rouge:
                    results["metrics"]["ROUGE"] = {}
                    rougeCalculator = RougeCalculator(source_text, summary_text)
                    if "ROUGE-1" in rouge_variants:
                        results["metrics"]["ROUGE"]["ROUGE-1"] = rougeCalculator.compute_rouge_n(1)
                    if "ROUGE-2" in rouge_variants:
                        results["metrics"]["ROUGE"]["ROUGE-2"] = rougeCalculator.compute_rouge_n(2)
                    if "ROUGE-L" in rouge_variants:
                        results["metrics"]["ROUGE"]["ROUGE-L"] = rougeCalculator.compute_rouge_l()

                # Compute BLEU
                if compute_bleu_check:
                    results["metrics"]["BLEU"] = BleuCalculator(source_text, summary_text).compute_bleu(bleu_max_n)

                # Compute ChrF
                if compute_chrf_check:
                    results["metrics"]["ChrF"] = ChrfCalculator(source_text, summary_text).compute_chrf(chrf_max_n)

                # Compute BERTScore
                if compute_bertscore_check:
                    with st.spinner(f"Loading {bert_model} and computing BERTScore..."):
                        results["metrics"]["BERTScore"] = BertScoreCalculator(source_text, summary_text, bert_model).compute_bert_score()

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

# ===== N-GRAM TAB =====
with tab_ngram:
    st.markdown("---")
    st.subheader("📊 N-gram Evaluation")
    df_ng, ng_start, ng_end = _batch_upload_and_range("ngram")
    if df_ng is not None:
        st.markdown("---")
        st.subheader("⚙️ Select Metrics")
        col1, col2 = st.columns(2)
        with col1:
            b_r1  = st.checkbox("ROUGE-1", True, key="ng_r1")
            b_r2  = st.checkbox("ROUGE-2", True, key="ng_r2")
            b_rl  = st.checkbox("ROUGE-L", True, key="ng_rl")
        with col2:
            b_bleu   = st.checkbox("BLEU",  True, key="ng_bleu")
            b_chrf   = st.checkbox("ChrF",  True, key="ng_chrf")
            b_f1     = st.checkbox("Token F1", True, key="ng_f1",
                                   help="Set-intersection F1: each unique word counted once. "
                                        "Matches the TRUE paper baseline. Differs from ROUGE-1 "
                                        "which clips by word frequency.")
            b_novel  = st.checkbox("Novel N-grams", True, key="ng_novel",
                                   help="Fraction of summary unigrams / bigrams / trigrams absent "
                                        "from the source. Higher = more abstractive / higher hallucination risk. "
                                        "Baseline from Aharoni et al. 2023 (multilingual summarisation).")

        if st.button("🚀 Evaluate", type="primary", use_container_width=True, key="run_ngram"):
            df_eval = df_ng.iloc[ng_start:ng_end].copy()
            results_list = []
            progress = st.progress(0)
            for n, (idx, row) in enumerate(df_eval.iterrows()):
                progress.progress((n + 1) / len(df_eval))
                p, h = str(row['premise']), str(row['hypothesis'])
                res = {'index': idx,
                       'source_length': len(p.split()),
                       'summary_length': len(h.split()),
                       'compression_ratio': round(len(h.split()) / max(len(p.split()), 1), 4)}
                rc = RougeCalculator(p, h)
                if b_r1:
                    r = rc.compute_rouge_n(1); res.update({'rouge1_precision': r['precision'], 'rouge1_recall': r['recall'], 'rouge1_f1': r['f1']})
                if b_r2:
                    r = rc.compute_rouge_n(2); res.update({'rouge2_precision': r['precision'], 'rouge2_recall': r['recall'], 'rouge2_f1': r['f1']})
                if b_rl:
                    r = rc.compute_rouge_l(); res.update({'rougel_precision': r['precision'], 'rougel_recall': r['recall'], 'rougel_f1': r['f1']})
                if b_bleu:
                    r = BleuCalculator(p, h).compute_bleu(4)
                    res['bleu'] = r['bleu']; res['bleu_bp'] = r['brevity_penalty']
                    for k, v in r['scores'].items(): res[k] = v
                if b_chrf:
                    res['chrf'] = ChrfCalculator(p, h).compute_chrf(6)['chrf']
                if b_f1:
                    r = TokenF1Calculator(p, h).compute()
                    res.update({'tokenf1_precision': r['precision'], 'tokenf1_recall': r['recall'], 'tokenf1_f1': r['f1']})
                if b_novel:
                    r = NovelNgramCalculator(p, h).compute()
                    res.update({'novel_unigram_pct': r['novel_unigram_pct'], 'novel_bigram_pct': r['novel_bigram_pct'], 'novel_trigram_pct': r['novel_trigram_pct']})
                results_list.append(res)
            progress.empty()
            _render_batch_results(pd.DataFrame(results_list), df_eval, "ngram_results")

# ===== SEMANTIC TAB =====
with tab_semantic:
    st.markdown("---")
    st.subheader("📊 Semantic Evaluation")
    df_sem, sem_start, sem_end = _batch_upload_and_range("semantic")
    if df_sem is not None:
        st.markdown("---")
        st.subheader("⚙️ BERTScore Configuration")
        sem_inference_type = st.radio(
            "Inference type:",
            ["Local", "Remote API"],
            horizontal=True,
            key="sem_inference_type",
            help="Local: runs bert-score on this machine. Remote API: calls inference server (GPU).",
        )
        if sem_inference_type == "Remote API":
            sem_api_url = st.text_input("API URL", value="http://localhost:8080", key="sem_api_url")
            if st.button("🏥 Check API Health", key="sem_health_check"):
                import requests as _req
                try:
                    resp = _req.get(f"{sem_api_url}/health", timeout=5)
                    if resp.status_code == 200:
                        h = resp.json()
                        if h.get("bertscore_ready"):
                            st.success("✅ BERTScore is ready on the remote API")
                        else:
                            st.error("❌ BERTScore not ready on remote API (bert-score library may not be installed on server)")
                    else:
                        st.error(f"❌ API returned status {resp.status_code}")
                except Exception as _e:
                    st.error(f"❌ Cannot reach API: {_e}")
        else:
            if not BERTSCORE_AVAILABLE:
                st.error("⚠️ BERTScore not available locally. Install: `pip install bert-score`")
        bert_model_batch = st.selectbox(
            "Hebrew BERT Model:",
            ["avichr/heBERT", "onlplab/alephbert-base", "dicta-il/neodictabert"],
            key="sem_bert_model",
        )
        if sem_inference_type == "Local":
            st.warning("⚠️ BERTScore loads a BERT model (~500MB) and is slow for large batches.")
        if st.button("🚀 Evaluate", type="primary", use_container_width=True, key="run_semantic"):
            if sem_inference_type == "Local" and not BERTSCORE_AVAILABLE:
                st.error("⚠️ BERTScore not available locally. Install: `pip install bert-score`")
            else:
                import requests as _requests
                df_eval = df_sem.iloc[sem_start:sem_end].copy()
                results_list = []
                progress = st.progress(0)
                for n, (idx, row) in enumerate(df_eval.iterrows()):
                    progress.progress((n + 1) / len(df_eval))
                    p, h = str(row['premise']), str(row['hypothesis'])
                    if sem_inference_type == "Remote API":
                        try:
                            resp = _requests.post(
                                f"{sem_api_url}/bertscore/predict",
                                json={"reference": p, "hypothesis": h, "model_name": bert_model_batch},
                                timeout=60,
                            )
                            r = resp.json() if resp.status_code == 200 else {"error": resp.text}
                        except Exception as _api_err:
                            r = {"error": str(_api_err)}
                    else:
                        r = BertScoreCalculator(p, h, bert_model_batch).compute_bert_score()
                    res = {'index': idx}
                    if 'error' not in r:
                        res.update({'bertscore_precision': r['precision'], 'bertscore_recall': r['recall'], 'bertscore_f1': r['f1']})
                    results_list.append(res)
                progress.empty()
                _render_batch_results(pd.DataFrame(results_list), df_eval, "semantic_results")

# ===== HALLUCINATION METRICS TAB =====
with tab_hallucination:
    from metrics.misc.hallucinations_metrics import (
        ENFS_EXPLANATION, ENTITY_LABELS,
        load_ner_pipeline, extract_entities, calculate_enfs, build_download_rows,
    )

    st.markdown("---")
    st.subheader("🎯 ENFS% — Entity Non-Factual Source")

    with st.expander("ℹ️ What is ENFS%?", expanded=False):
        st.markdown(ENFS_EXPLANATION)

    try:
        from transformers import pipeline as _hf_pipeline  # noqa: F401
        NER_AVAILABLE = True
    except ImportError:
        NER_AVAILABLE = False
        st.error("⚠️ transformers library required. Install: `pip install transformers`")

    if NER_AVAILABLE:
        halluc_file = st.file_uploader(
            "Upload CSV with premise and hypothesis columns",
            type=["csv"],
            key="halluc_csv",
        )

        col1, col2 = st.columns(2)
        with col1:
            ner_model_choice = st.selectbox(
                "NER Model:",
                list(ENTITY_LABELS.keys()),
                key="ner_model_select",
            )
        with col2:
            ner_threshold = st.slider(
                "NER Confidence Threshold:",
                min_value=0.3, max_value=0.9, value=0.7, step=0.05,
                key="ner_threshold",
            )

        if halluc_file is not None:
            try:
                df_halluc = pd.read_csv(halluc_file)
                st.success(f"✅ Loaded {len(df_halluc)} records")

                if not {'premise', 'hypothesis'}.issubset(df_halluc.columns):
                    st.error("❌ CSV must contain 'premise' and 'hypothesis' columns")
                else:
                    if 'label' in df_halluc.columns:
                        filter_label = st.multiselect(
                            "Filter by label (optional):",
                            options=df_halluc['label'].unique().tolist(),
                            key="halluc_label_filter",
                        )
                        if filter_label:
                            df_halluc = df_halluc[df_halluc['label'].isin(filter_label)]
                            st.info(f"📊 Filtered to {len(df_halluc)} records")

                    with st.expander("🔍 Dataset Preview", expanded=False):
                        st.dataframe(df_halluc, use_container_width=True, height=300)

                    max_samples = len(df_halluc)
                    sample_size = int(st.number_input(
                        "Number of samples to evaluate:",
                        min_value=1, max_value=max_samples,
                        value=min(50, max_samples), step=1,
                        key="halluc_sample_size",
                    ))

                    if st.button("🚀 Calculate ENFS%", type="primary", use_container_width=True):
                        with st.spinner(f"Loading {ner_model_choice}..."):
                            ner_pipe = load_ner_pipeline(ner_model_choice)

                        df_eval = df_halluc.head(sample_size).copy()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results_list = []

                        for i, (idx, row) in enumerate(df_eval.iterrows()):
                            status_text.text(f"Processing {i + 1}/{sample_size}...")
                            premise_ents = extract_entities(str(row['premise']), ner_pipe, ner_threshold)
                            hyp_ents = extract_entities(str(row['hypothesis']), ner_pipe, ner_threshold)
                            enfs, total_ents, halluc_count, halluc_ents = calculate_enfs(premise_ents, hyp_ents)
                            results_list.append({
                                'index': i,
                                'enfs_percent': enfs,
                                'total_entities': total_ents,
                                'hallucinated_entities': halluc_count,
                                'premise_entities': premise_ents,
                                'hypothesis_entities': hyp_ents,
                                'hallucinated_list': halluc_ents,
                            })
                            progress_bar.progress((i + 1) / sample_size)

                        progress_bar.empty()
                        status_text.empty()
                        st.session_state['halluc_results_list'] = results_list
                        st.session_state['halluc_df_eval'] = df_eval

                    if st.session_state.get('halluc_results_list'):
                        results_list = st.session_state['halluc_results_list']
                        df_eval = st.session_state['halluc_df_eval']

                        st.success("✅ ENFS% calculation complete!")
                        st.markdown("---")
                        st.subheader("📊 Results Summary")

                        valid_enfs = [r['enfs_percent'] for r in results_list if r['enfs_percent'] is not None]
                        total_hyp_ents = sum(r['total_entities'] for r in results_list)
                        total_halluc = sum(r['hallucinated_entities'] for r in results_list)

                        if valid_enfs:
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Mean ENFS%", f"{np.mean(valid_enfs):.2f}%")
                            with col2:
                                st.metric("Median ENFS%", f"{np.median(valid_enfs):.2f}%")
                            with col3:
                                st.metric("Total Entities (Premise)", sum(len(r['premise_entities']) for r in results_list))
                            with col4:
                                st.metric("Total Entities (Summary)", total_hyp_ents)
                            with col5:
                                st.metric("Samples w/ Hallucinations", f"{sum(1 for e in valid_enfs if e > 0)}/{len(valid_enfs)}")

                            if total_hyp_ents > 0:
                                st.markdown(f"**Total Hallucinated Entities: {total_halluc}** ({total_halluc}/{total_hyp_ents} = {total_halluc/total_hyp_ents*100:.1f}% overall)")

                            st.markdown("---")
                            st.subheader("📋 Detailed Results")
                            results_data = []
                            for r in results_list:
                                row_data = df_eval.iloc[r['index']].to_dict()
                                row_data['enfs_percent'] = f"{r['enfs_percent']:.1f}%" if r['enfs_percent'] is not None else "N/A"
                                row_data['total_entities_summary'] = r['total_entities']
                                row_data['total_entities_premise'] = len(r['premise_entities'])
                                row_data['hallucinated_count'] = r['hallucinated_entities']
                                results_data.append(row_data)

                            results_display_df = pd.DataFrame(results_data)
                            orig_cols = [c for c in df_eval.columns if c in results_display_df.columns]
                            metric_cols = ['enfs_percent', 'total_entities_summary', 'total_entities_premise', 'hallucinated_count']
                            st.dataframe(results_display_df[orig_cols + metric_cols], use_container_width=True, height=400)

                            st.markdown("---")
                            st.subheader("🔍 Entity Details")
                            selected_row = st.selectbox(
                                "Select row to inspect:",
                                range(len(results_list)),
                                format_func=lambda x: (
                                    f"Row {results_list[x]['index']} (ENFS%: {results_list[x]['enfs_percent']:.1f}%)"
                                    if results_list[x]['enfs_percent'] is not None
                                    else f"Row {results_list[x]['index']} (No entities)"
                                ),
                                key="halluc_selected_row",
                            )

                            sel = results_list[selected_row]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Entities in Source (Premise):**")
                                for ent in sel['premise_entities']:
                                    st.markdown(f"- `{ent['text']}` ({ent['label']}) — {ent['score']:.2f}")
                                if not sel['premise_entities']:
                                    st.info("No entities found")
                            with col2:
                                st.markdown("**Entities in Summary (Hypothesis):**")
                                for ent in sel['hypothesis_entities']:
                                    marker = "🚨" if ent in sel['hallucinated_list'] else "✅"
                                    st.markdown(f"{marker} `{ent['text']}` ({ent['label']}) — {ent['score']:.2f}")
                                if not sel['hypothesis_entities']:
                                    st.info("No entities found")

                            if sel['hallucinated_list']:
                                st.warning(f"Hallucinated: {', '.join(e['text'] for e in sel['hallucinated_list'])}")
                            else:
                                st.success("✅ No hallucinations detected")

                            with st.expander("📖 View Full Texts", expanded=False):
                                st.text_area("Source:", value=df_eval.iloc[selected_row]['premise'], height=200, disabled=True, key="hal_src")
                                st.text_area("Summary:", value=df_eval.iloc[selected_row]['hypothesis'], height=150, disabled=True, key="hal_hyp")

                            st.markdown("---")
                            download_df = pd.DataFrame(build_download_rows(results_list, df_eval))
                            st.download_button(
                                "📥 Download ENFS% Results (CSV)",
                                download_df.to_csv(index=False).encode('utf-8'),
                                "enfs_results.csv",
                                "text/csv",
                            )
                        else:
                            st.warning("⚠️ No valid ENFS% scores (all summaries have 0 entities)")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("📁 Upload a CSV with `premise`, `hypothesis` (and optional `label`) columns.")

# Help section
with st.expander("ℹ️ Metric Explanations"):
    st.markdown(ALL_METRICS_EXPLANATION)

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
                
                label_binary = label_to_binary(labels_true)
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

# ===== COMPARE METRICS TAB =====
with tab_compare:
    st.markdown("---")
    st.subheader("📐 Compare Evaluation Metrics")
    st.markdown("""
    Upload a CSV with a **label column** and any metric score columns.
    This tab measures how well each metric correlates with the ground-truth label —
    helping you decide which metric best predicts factual correctness.
    """)

    compare_file = st.file_uploader(
        "Upload CSV with label + metric score columns",
        type=["csv"],
        key="compare_csv",
        help="Required: label column. Optional: rouge1_f1, rouge2_f1, rougel_f1, bleu, chrf, bertscore_f1, prob_entailment, or any numeric score column."
    )

    if compare_file is not None:
        try:
            df_compare = pd.read_csv(compare_file)
            st.success(f"✅ Loaded {len(df_compare)} records")

            if 'label' not in df_compare.columns:
                st.error("❌ CSV must contain a 'label' column (entailment/contradiction or 0/1)")
            else:
                # Auto-detect numeric columns that could be metrics
                non_metric_columns = {'label', 'premise', 'hypothesis', 'index', 'predicted_label', 'true_label'}
                candidate_columns = [
                    col for col in df_compare.columns
                    if col not in non_metric_columns and pd.api.types.is_numeric_dtype(df_compare[col])
                ]

                if not candidate_columns:
                    st.warning("⚠️ No numeric metric columns found besides 'label'.")
                else:
                    st.markdown("---")
                    st.subheader("⚙️ Configuration")

                    col1, col2 = st.columns(2)

                    with col1:
                        selected_metrics = st.multiselect(
                            "Select metric columns to compare:",
                            options=candidate_columns,
                            default=candidate_columns,
                            help="Choose which numeric columns to treat as metric scores"
                        )

                    with col2:
                        comparison_method = st.radio(
                            "Comparison method:",
                            ["Kendall τ", "Point-biserial correlation"],
                            help="Kendall τ — rank-based. Point-biserial — correct for binary labels vs continuous scores."
                        )

                    if st.button("📐 Run Comparison", type="primary", use_container_width=True):
                        label_binary = label_to_binary(df_compare['label'])
                        valid_indices = [i for i, lbl in enumerate(label_binary) if lbl is not None]

                        if len(valid_indices) < 2:
                            st.error("❌ Need at least 2 samples with valid binary labels (0/1 or entailment/contradiction).")
                        else:
                            clean_labels = [label_binary[i] for i in valid_indices]

                            comparison_results = []
                            for column_name in selected_metrics:
                                scores = [df_compare.iloc[i][column_name] for i in valid_indices]
                                if comparison_method == "Kendall τ":
                                    result = KendallTauCalculator(scores, clean_labels, column_name).compute()
                                else:
                                    result = PointBiserialCalculator(scores, clean_labels, column_name).compute()
                                comparison_results.append(result)

                            # Sort by absolute statistic descending
                            comparison_results.sort(key=lambda r: abs(r.statistic) if not math.isnan(r.statistic) else -1, reverse=True)

                            st.markdown("---")
                            stat_label = "τ" if comparison_method == "Kendall τ" else "r"
                            st.subheader(f"📊 Results — {comparison_method}")

                            cols = st.columns(min(3, len(comparison_results)))
                            for idx, result in enumerate(comparison_results):
                                delta_color = "normal" if result.statistic > 0.3 else ("inverse" if result.statistic < -0.1 else "off")
                                with cols[idx % 3]:
                                    st.metric(
                                        result.metric_name,
                                        f"{stat_label} = {result.statistic:.4f}",
                                        f"p = {result.p_value:.4f}  {'✅' if result.is_significant else ''}",
                                        delta_color=delta_color,
                                        help=result.interpretation
                                    )

                            # Summary table
                            st.markdown("---")
                            st.subheader("📋 Summary Table")
                            summary_rows = [
                                {
                                    "Metric": r.metric_name,
                                    stat_label: round(r.statistic, 4),
                                    "p-value": round(r.p_value, 4),
                                    "Significant (p<0.05)": "✅" if r.is_significant else "❌",
                                    "Interpretation": r.interpretation,
                                }
                                for r in comparison_results
                            ]
                            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                            csv_out = pd.DataFrame(summary_rows).to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Comparison Results (CSV)",
                                csv_out,
                                "metric_comparison.csv",
                                "text/csv"
                            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("📁 Upload a CSV file to start metric comparison")
        st.markdown("""
        **Expected CSV format:**
        ```
        label, rouge1_f1, bleu, chrf, prob_entailment
        entailment, 0.45, 0.12, 38.2, 0.91
        contradiction, 0.21, 0.05, 18.7, 0.23
        ```

        **Any numeric column** is automatically detected as a candidate metric.
        The label column must be named `label` and contain `entailment/contradiction` or `0/1`.
        """)

with tab_qafacteval:
    st.markdown("---")
    st.subheader("🔍 QAFactEval — Question-Answering Based Factual Evaluation")
    st.markdown("""
    Evaluates factual consistency by generating questions from the summary and checking if the source article can answer them.
    """)

    qa_file = st.file_uploader(
        "Upload CSV (premise, hypothesis columns required; label optional)",
        type=["csv"],
        key="qa_csv",
    )

    if qa_file is not None:

        try:
            df_qa = pd.read_csv(qa_file)
            st.success(f"✅ Loaded {len(df_qa)} records")

            if not {'premise', 'hypothesis'}.issubset(df_qa.columns):
                st.error("❌ CSV must contain 'premise' and 'hypothesis' columns")
            else:
                max_qa = len(df_qa)
                col_s, col_e = st.columns(2)
                with col_s:
                    qa_start = int(st.number_input("Start index:", min_value=0, max_value=max_qa - 1, value=0, step=1, key="qa_start"))
                with col_e:
                    qa_end = int(st.number_input("End index:", min_value=qa_start + 1, max_value=max_qa, value=min(qa_start + 10, max_qa), step=1, key="qa_end"))

                st.markdown("---")
                st.subheader("⚙️ Pipeline Configuration")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Step 1: Answer Selection**")
                    answer_backend = st.radio(
                        "Backend:",
                        ["DictaBERT Shallow NP (paper best)", "DictaBERT Max NP", "OpenAI"],
                        key="qa_selector",
                        help="Shallow NP: head token only — closest to the paper's best-performing NP Chunks strategy (77.5). Max NP: head + modifiers — slightly lower (75.7). OpenAI: GPT-based extraction.",
                    )

                with col2:
                    st.markdown("**Step 2: Question Generation**")
                    qg_backend = st.radio("Backend:", ["OpenAI", "HuggingFace"], key="qa_qg")
                    qg_model = None
                    if qg_backend == "HuggingFace":
                        qg_model = st.text_input("HF model name:", value="nielsr/MixQG-large", key="qa_qg_model")

                with col3:
                    st.markdown("**Step 3: Question Answering**")
                    qa_backend = st.radio("Backend:", ["XLM-RoBERTa (HF)", "OpenAI"], key="qa_qa")

                st.markdown("---")
                st.markdown("**Step 3a: IsAnsweredSumm — F1 gate threshold**")
                isanswered_threshold = st.slider(
                    "F1(NP chunk, summary answer) threshold",
                    min_value=0.0, max_value=1.0, value=0.60, step=0.05,
                    key="qa_isanswered_threshold",
                    help="Questions where the summary answer's token F1 vs the NP chunk is below this threshold are discarded. Paper default: 0.60.",
                )

                st.markdown("---")
                st.markdown("**Step 4: Answer Comparison**")
                cmp_col1, cmp_col2 = st.columns([1, 2])
                with cmp_col1:
                    cmp_method = st.radio("Method:", ["Token F1", "Exact Match", "GPT Judge"], key="qa_cmp",
                                          help="Token F1: lexical overlap. Exact Match: 1.0 only when answers match exactly. GPT Judge: LLM semantic equivalence.")
                with cmp_col2:
                    if cmp_method == "GPT Judge":
                        st.info("GPT will receive the question, both answers, and both full texts to judge equivalence.")
                    elif cmp_method == "Exact Match":
                        st.info("Answers are normalized (lowercase, strip punctuation) before comparison.")

                needs_openai = (answer_backend == "OpenAI" or qg_backend == "OpenAI"
                                or qa_backend == "OpenAI" or cmp_method == "GPT Judge")

                openai_model = "gpt-4o-mini"
                if needs_openai:
                    st.markdown("---")
                    openai_model = st.selectbox("OpenAI Model:", ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"], key="qa_openai_model")

                max_workers = int(st.number_input(
                    "Parallel workers:",
                    min_value=1, max_value=8, value=4, step=1,
                    key="qa_workers",
                    help="Samples evaluated simultaneously. Higher = faster but uses more memory / API quota.",
                ))

                # ── Run / Stop buttons ────────────────────────────────────────────
                run_col, stop_col = st.columns(2)
                with run_col:
                    run_clicked = st.button(
                        "🔍 Run QAFactEval",
                        type="primary",
                        use_container_width=True,
                        disabled=st.session_state.get('qa_running', False),
                    )
                with stop_col:
                    stop_clicked = st.button(
                        "⏹ Stop",
                        type="secondary",
                        use_container_width=True,
                        disabled=not st.session_state.get('qa_running', False),
                    )

                if stop_clicked:
                    stop_ev = st.session_state.get('qa_stop_event')
                    if stop_ev is not None:
                        stop_ev.set()
                    st.session_state['qa_running'] = False
                    st.session_state['qa_stop_requested'] = True

                if run_clicked:
                    from metrics.qa_facteval import (
                        PipelineConfig, QaFactEvalPipeline,
                        DictaBertAnswerSelector, DictaBertShallowNPSelector, OpenAiAnswerSelector,
                        OpenAiQuestionGenerator, HfQuestionGenerator,
                        HfQuestionAnswerer, OpenAiQuestionAnswerer,
                        TokenF1Comparator, ExactMatchComparator, OpenAiAnswerComparator,
                    )

                    openai_client = None
                    if needs_openai:
                        from openai import OpenAI
                        openai_api_key = os.getenv("OPENAI_API_KEY", "")
                        if not openai_api_key:
                            st.error("❌ OPENAI_API_KEY not found in .env file.")
                            st.stop()
                        openai_client = OpenAI(api_key=openai_api_key)

                    if answer_backend == "DictaBERT Shallow NP (paper best)":
                        selector = DictaBertShallowNPSelector()
                    elif answer_backend == "DictaBERT Max NP":
                        selector = DictaBertAnswerSelector()
                    else:
                        selector = OpenAiAnswerSelector(openai_client, openai_model)
                    generator = (
                        OpenAiQuestionGenerator(openai_client, openai_model)
                        if qg_backend == "OpenAI"
                        else HfQuestionGenerator(qg_model)
                    )
                    answerer = (
                        HfQuestionAnswerer()
                        if qa_backend == "XLM-RoBERTa (HF)"
                        else OpenAiQuestionAnswerer(openai_client, openai_model)
                    )
                    if cmp_method == "GPT Judge":
                        comparator = OpenAiAnswerComparator(openai_client, openai_model)
                    elif cmp_method == "Exact Match":
                        comparator = ExactMatchComparator()
                    else:
                        comparator = TokenF1Comparator()

                    pipeline_obj = QaFactEvalPipeline(PipelineConfig(
                        answer_selector=selector,
                        question_generator=generator,
                        question_answerer=answerer,
                        answer_comparator=comparator,
                        isanswered_f1_threshold=isanswered_threshold,
                    ))
                    df_slice_new = df_qa.iloc[qa_start:qa_end].copy()
                    pairs_new = [
                        (str(row['hypothesis']), str(row['premise']))
                        for _, row in df_slice_new.iterrows()
                    ]
                    with st.spinner("Warming up models..."):
                        pipeline_obj.warm_up()

                    stop_event_new = threading.Event()
                    results_queue_new = queue.Queue()

                    def _background_run(
                        pipeline_ref,
                        pairs_ref,
                        workers_count,
                        stop_event_ref,
                        result_queue_ref,
                    ):
                        pipeline_ref.run_batch(
                            pairs_ref,
                            max_workers=workers_count,
                            stop_event=stop_event_ref,
                            result_queue=result_queue_ref,
                        )

                    bg_thread = threading.Thread(
                        target=_background_run,
                        args=(pipeline_obj, pairs_new, max_workers, stop_event_new, results_queue_new),
                        daemon=True,
                    )

                    st.session_state['qa_pipeline'] = pipeline_obj
                    st.session_state['qa_pairs'] = pairs_new
                    st.session_state['qa_df_slice'] = df_slice_new
                    st.session_state['qa_partial_results'] = []
                    st.session_state['qa_running'] = True
                    st.session_state['qa_stop_requested'] = False
                    st.session_state['qa_stop_event'] = stop_event_new
                    st.session_state['qa_results_queue'] = results_queue_new
                    st.session_state['qa_bg_thread'] = bg_thread
                    st.session_state['qa_start_offset'] = qa_start
                    st.session_state['qa_cmp_method'] = cmp_method
                    st.session_state['qa_run_config'] = {
                        'file_name': qa_file.name,
                        'total_size': len(df_qa),
                        'answer_backend': answer_backend,
                        'qg_backend': qg_backend,
                        'qg_model': qg_model,
                        'qa_backend': qa_backend,
                        'cmp_method': cmp_method,
                        'isanswered_threshold': isanswered_threshold,
                        'openai_model': openai_model if needs_openai else 'N/A',
                        'qa_start': qa_start,
                        'qa_end': qa_end,
                        'max_workers': max_workers,
                    }
                    bg_thread.start()
                    st.rerun()

                # ── Live progress: drain queue, update bar, poll until done ─────
                if st.session_state.get('qa_running', False):
                    results_q = st.session_state['qa_results_queue']
                    df_s = st.session_state['qa_df_slice']
                    start_off = st.session_state['qa_start_offset']
                    total_samples = len(st.session_state['qa_pairs'])

                    # Drain all results that arrived since the last rerun
                    while not results_q.empty():
                        raw_item = results_q.get_nowait()
                        sample_idx = raw_item['index']
                        row = df_s.iloc[sample_idx]
                        if raw_item.get('error'):
                            result_dict = {
                                'index': start_off + sample_idx,
                                'error': raw_item['error'],
                                'f1_score': 0.0,
                                'qa_answers': [],
                            }
                        else:
                            qa_result = raw_item['result']
                            answerable_count = sum(1 for answer in qa_result.qa_answers if answer.is_answerable)
                            result_dict = {
                                'index': start_off + sample_idx,
                                'hypothesis': str(row['hypothesis']),
                                'premise': str(row['premise']),
                                'label': str(row['label']) if 'label' in row.index else '',
                                'questions_generated': qa_result.questions_generated,
                                'questions_after_filter': qa_result.questions_after_filter,
                                'answerable': answerable_count,
                                'f1_score': round(qa_result.f1_score, 4),
                                'qa_answers': [
                                    {
                                        'np_chunk': answer.np_chunk,
                                        'question': answer.question,
                                        'summary_answer': answer.summary_answer,
                                        'source_answer': answer.source_answer,
                                        'is_answerable': answer.is_answerable,
                                        'answer_f1': round(answer.answer_f1, 4),
                                    }
                                    for answer in qa_result.qa_answers
                                ],
                            }
                        st.session_state['qa_partial_results'].append(result_dict)

                    completed_count = len(st.session_state['qa_partial_results'])
                    st.progress(
                        completed_count / total_samples,
                        text=f"Processed {completed_count} / {total_samples} samples  ({max_workers} workers)",
                    )

                    bg_thread = st.session_state.get('qa_bg_thread')
                    if bg_thread and not bg_thread.is_alive():
                        # Background thread finished (completed or stopped)
                        st.session_state['qa_running'] = False
                        st.rerun()
                    else:
                        # Poll again in 0.5 s
                        time.sleep(0.5)
                        st.rerun()

                # ── Results (shown after run completes or is stopped) ─────────────
                if (not st.session_state.get('qa_running', False)
                        and st.session_state.get('qa_partial_results')):

                    # Sort by original input index (parallel completion is unordered)
                    all_qa_results = sorted(
                        st.session_state['qa_partial_results'],
                        key=lambda result_item: result_item['index'],
                    )
                    cfg = st.session_state.get('qa_run_config', {})
                    n_evaluated = len(all_qa_results)

                    if st.session_state.get('qa_stop_requested', False):
                        st.warning(f"⏹ Stopped — {n_evaluated} / {cfg.get('total_size', '?')} samples evaluated.")
                    else:
                        st.success(f"✅ Done! Processed {n_evaluated} samples.")

                    st.session_state['qafacteval_log'] = all_qa_results
                    st.session_state['qafacteval_config_md'] = _build_run_config_md(
                        "QAFactEval",
                        {"name": cfg.get('file_name', ''), "total_size": cfg.get('total_size', '?'), "evaluated": n_evaluated},
                        [
                            ("Row Range", f"{cfg.get('qa_start')}–{cfg.get('qa_start', 0) + n_evaluated - 1} ({n_evaluated} samples)"),
                            ("Step 1 — Answer Selection", cfg.get('answer_backend', '')),
                            ("Step 2 — Question Generation", cfg.get('qg_backend', '') + (f" ({cfg.get('qg_model')})" if cfg.get('qg_backend') == "HuggingFace" else "")),
                            ("Step 3 — Question Answering", cfg.get('qa_backend', '')),
                            ("Step 4 — Answer Comparison", cfg.get('cmp_method', '')),
                            ("IsAnsweredSumm F1 Threshold", cfg.get('isanswered_threshold', '')),
                            ("OpenAI Model", cfg.get('openai_model', 'N/A')),
                            ("Parallel Workers", cfg.get('max_workers', '')),
                        ],
                    )

                    cmp_method_stored = cfg.get('cmp_method', '')
                    summary_rows = [{k: v for k, v in r.items() if k not in ('qa_answers', 'hypothesis', 'premise', 'label')} for r in all_qa_results]
                    result_df = pd.DataFrame(summary_rows)
                    mean_f1 = result_df['f1_score'].mean()

                    st.markdown("---")
                    score_label = {
                        "GPT Judge": "Mean QAFactEval Score (GPT Judge)",
                        "Exact Match": "Mean QAFactEval Score (Exact Match)",
                    }.get(cmp_method_stored, "Mean QAFactEval F1 Score")
                    st.metric(score_label, f"{mean_f1:.4f}")
                    st.dataframe(result_df, use_container_width=True)

                    download_csv_json(
                        csv_bytes=result_df.to_csv(index=False).encode('utf-8'),
                        json_bytes=json.dumps(all_qa_results, ensure_ascii=False, indent=2).encode('utf-8'),
                        base_filename="qafacteval_results",
                    )

                if st.session_state.get('qafacteval_log'):
                    log = st.session_state['qafacteval_log']
                    st.markdown("---")
                    summary_rows_dl = [{k: v for k, v in r.items() if k not in ('qa_answers', 'hypothesis', 'premise', 'label')} for r in log]
                    download_csv_json(
                        csv_bytes=pd.DataFrame(summary_rows_dl).to_csv(index=False).encode('utf-8'),
                        json_bytes=json.dumps(log, ensure_ascii=False, indent=2).encode('utf-8'),
                        base_filename="qafacteval_results",
                        csv_key="dl_csv_persistent",
                        json_key="dl_json_persistent",
                    )
                    if st.session_state.get('qafacteval_config_md'):
                        st.download_button("📋 Download Config (MD)", st.session_state['qafacteval_config_md'].encode(), "qafacteval_config.md", "text/markdown", key="dl_md_qafacteval")
                    st.subheader("🔎 Trace Viewer")
                    selected_sample = st.selectbox(
                        "Select sample to inspect:",
                        range(len(log)),
                        format_func=lambda x: f"Sample {log[x]['index']} — score: {log[x].get('score', 'N/A')}",
                        key="qa_trace_select",
                    )
                    entry = log[selected_sample]

                    with st.expander("📄 Texts", expanded=False):
                        if entry.get('label'):
                            st.markdown(f"**Label:** `{entry['label']}`")
                        st.markdown("**Hypothesis (summary):**")
                        st.write(entry.get('hypothesis', ''))
                        st.markdown("**Premise (source):**")
                        st.write(entry.get('premise', ''))

                    if entry.get('error'):
                        st.error(f"Pipeline error: {entry['error']}")
                    else:
                        st.markdown(f"Questions generated: **{entry['questions_generated']}** → after IsAnsweredSumm filter: **{entry['questions_after_filter']}** → answerable: **{entry['answerable']}** → F1 score: **{entry.get('f1_score', 0):.4f}**")
                        for j, qa in enumerate(entry.get('qa_answers', []), 1):
                            status = "✅" if qa['is_answerable'] else "❌"
                            f1 = qa.get('answer_f1', 0.0)
                            f1_color = "🟢" if f1 >= 0.5 else ("🟡" if f1 >= 0.2 else "🔴")
                            with st.expander(f"Q{j}: {qa['question'][:70]}...  {status} F1={f1:.2f} {f1_color}", expanded=False):
                                st.markdown(f"**NP chunk (answer seed):** `{qa['np_chunk']}`")
                                st.markdown(f"**Question:** {qa['question']}")
                                st.markdown("---")
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.markdown("**Step 3b — Summary QA answer** *(used for IsAnsweredSumm filter only)*")
                                    st.info(qa.get('summary_answer') or '—')
                                with col_b:
                                    st.markdown("**Step 4 — Source QA answer** *(scored against NP chunk above)*")
                                    st.info(qa.get('source_answer') or '—')
                                with col_c:
                                    st.metric("Score", f"{f1:.4f}", help="overlap(NP chunk, source answer). Per paper: the selected answer (NP chunk) is the reference, the source QA output is the candidate.")
                                    st.metric("Answerable", status)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("📁 Upload a CSV file to start QAFactEval evaluation")
        st.markdown("""
        **Expected CSV format:**
        ```
        premise, hypothesis, label
        source article text..., generated summary..., entailment
        ```
        - `premise` — the source article
        - `hypothesis` — the generated summary
        - `label` — optional, for reference only
        """)

# ===== SUMMAC TAB =====
with tab_summac:
    st.markdown("---")
    st.subheader("🧠 SummaC — NLI-Based Factual Consistency")
    st.markdown("Scores factual consistency by running an NLI model on every (doc sentence, summary sentence) pair and aggregating the entailment scores.")

    summac_file = st.file_uploader(
        "Upload CSV (premise, hypothesis columns required; label optional)",
        type=["csv"], key="summac_csv",
    )

    if summac_file is not None:
        try:
            df_sc = pd.read_csv(summac_file)
            st.success(f"✅ Loaded {len(df_sc)} records")

            if not {'premise', 'hypothesis'}.issubset(df_sc.columns):
                st.error("❌ CSV must contain 'premise' and 'hypothesis' columns")
            else:
                if 'label' in df_sc.columns:
                    available_labels = df_sc['label'].dropna().unique().tolist()
                    selected_labels = st.multiselect(
                        "Filter by label (optional):",
                        options=available_labels,
                        default=[],
                        key="sc_label_filter",
                    )
                    if selected_labels:
                        df_sc = df_sc[df_sc['label'].isin(selected_labels)].reset_index(drop=True)
                        st.info(f"📊 Filtered to {len(df_sc)} records")

                col_s, col_e = st.columns(2)
                with col_s:
                    sc_start = int(st.number_input("Start index:", min_value=0, max_value=len(df_sc) - 1, value=0, step=1, key="sc_start"))
                with col_e:
                    sc_end = int(st.number_input("End index:", min_value=sc_start + 1, max_value=len(df_sc), value=min(sc_start + 5, len(df_sc)), step=1, key="sc_end"))

                st.markdown("---")
                st.subheader("⚙️ Configuration")

                from metrics.summac import HfNliScorer
                sc_model = st.selectbox("NLI Model:", HfNliScorer.SUPPORTED_MODELS, key="sc_model")
                sc_threshold = st.slider("Decision threshold:", 0.0, 1.0, 0.5, 0.05, key="sc_threshold",
                                         help="Score ≥ threshold → consistent. Tune on your validation set.")
                sc_workers = int(st.number_input("Parallel workers:", 1, 4, 1, key="sc_workers"))

                sc_inference_type = st.radio(
                    "Inference type:",
                    ["Local", "Remote API"],
                    horizontal=True,
                    key="sc_inference_type",
                    help="Local: loads NLI model on this machine. Remote API: calls inference server on GPU.",
                )
                if sc_inference_type == "Remote API":
                    sc_api_url = st.text_input("API URL", value="http://localhost:8080", key="sc_api_url")
                    if st.button("🏥 Check API Health", key="sc_health_check"):
                        import requests as _req
                        try:
                            resp = _req.get(f"{sc_api_url}/health", timeout=5)
                            if resp.status_code == 200:
                                h = resp.json()
                                if h.get("summac_ready"):
                                    st.success("✅ SummaC is ready on the remote API")
                                else:
                                    st.error("❌ SummaC not ready on remote API")
                            else:
                                st.error(f"❌ API returned status {resp.status_code}")
                        except Exception as _e:
                            st.error(f"❌ Cannot reach API: {_e}")

                if st.button("▶️ Run SummaC", type="primary", use_container_width=True):
                    import requests as _requests
                    df_slice = df_sc.iloc[sc_start:sc_end].copy()
                    log = []

                    if sc_inference_type == "Remote API":
                        progress_bar = st.progress(0, text="Starting SummaC via API...")
                        for i, (_, row) in enumerate(df_slice.iterrows()):
                            try:
                                resp = _requests.post(
                                    f"{sc_api_url}/summac/predict",
                                    json={"source_document": str(row['premise']), "summary": str(row['hypothesis']), "model_name": sc_model},
                                    timeout=120,
                                )
                                if resp.status_code == 200:
                                    r = resp.json()
                                    log.append({
                                        "index": sc_start + i,
                                        "label": str(row.get('label', '')),
                                        "summary_score": r["summary_score"],
                                        "is_consistent": r["summary_score"] >= sc_threshold,
                                        "threshold": sc_threshold,
                                        "doc_sentences": r["doc_sentences"],
                                        "summary_sentences": r["summary_sentences"],
                                        "nli_matrix": r["nli_matrix"],
                                        "sentence_scores": r["sentence_scores"],
                                        "hypothesis": str(row['hypothesis']),
                                        "premise": str(row['premise']),
                                    })
                                else:
                                    log.append({"index": sc_start + i, "error": resp.text, "summary_score": 0.0})
                            except Exception as _e:
                                log.append({"index": sc_start + i, "error": str(_e), "summary_score": 0.0})
                            progress_bar.progress((i + 1) / len(df_slice), text=f"Scored {i + 1} / {len(df_slice)}")
                        progress_bar.empty()
                    else:
                        from metrics.summac import SummacFactory, serialize_result
                        pipeline_sc = SummacFactory.zero_shot(sc_model, sc_threshold)
                        pairs = [(str(r['hypothesis']), str(r['premise'])) for _, r in df_slice.iterrows()]
                        with st.spinner(f"Running SummaC on {len(pairs)} samples..."):
                            batch = pipeline_sc.score_batch(pairs, max_workers=sc_workers)
                        for i, item in enumerate(batch):
                            row = df_slice.iloc[i]
                            if item.get("error"):
                                log.append({"index": sc_start + i, "error": item["error"], "summary_score": 0.0})
                            else:
                                log.append(serialize_result(
                                    sc_start + i, str(row['premise']), str(row['hypothesis']),
                                    str(row.get('label', '')), item["result"],
                                ))

                    st.session_state['summac_log'] = log
                    st.session_state['summac_config_md'] = _build_run_config_md(
                        "SummaC",
                        {"name": summac_file.name, "total_size": len(df_sc), "evaluated": len(df_slice)},
                        [
                            ("Inference Type", sc_inference_type),
                            ("Variant", "Zero-Shot (col-max → mean)"),
                            ("NLI Model", sc_model),
                            ("Decision Threshold", sc_threshold),
                            ("Parallel Workers", sc_workers if sc_inference_type == "Local" else "N/A (API)"),
                            ("Row Range", f"{sc_start}–{sc_end - 1} ({len(df_slice)} samples)"),
                        ],
                    )
                    st.success("✅ Done!")

                    summary_df = pd.DataFrame([
                        {"index": r["index"], "label": r.get("label", ""), "score": r["summary_score"],
                         "consistent": r.get("is_consistent", ""), "error": r.get("error", "")}
                        for r in log
                    ])
                    mean_score = summary_df["score"].mean()
                    st.metric("Mean SummaC Score", f"{mean_score:.4f}")
                    st.dataframe(summary_df, use_container_width=True)

                    download_csv_json(
                        csv_bytes=summary_df.to_csv(index=False).encode(),
                        json_bytes=json.dumps(log, ensure_ascii=False, indent=2).encode(),
                        base_filename="summac_results",
                    )

                # ── Trace Viewer ──────────────────────────────────────────────
                if st.session_state.get('summac_log'):
                    log = st.session_state['summac_log']
                    st.markdown("---")
                    if st.session_state.get('summac_config_md'):
                        st.download_button("📋 Download Config (MD)", st.session_state['summac_config_md'].encode(), "summac_config.md", "text/markdown", key="dl_md_summac")
                    st.subheader("🔎 Trace Viewer")
                    selected = st.selectbox(
                        "Select sample:",
                        range(len(log)),
                        format_func=lambda x: f"Sample {log[x]['index']} — score: {log[x]['summary_score']:.4f}  {'✅' if log[x].get('is_consistent') else '❌'}",
                        key="sc_trace_select",
                    )
                    entry = log[selected]

                    if entry.get("error"):
                        st.error(f"Pipeline error:\n{entry['error']}")
                    else:
                        with st.expander("📄 Texts", expanded=False):
                            if entry.get("label"):
                                st.markdown(f"**Label:** `{entry['label']}`")
                            st.markdown("**Hypothesis (summary):**")
                            st.write(entry["hypothesis"])
                            st.markdown("**Premise (source):**")
                            st.write(entry["premise"])

                        st.markdown(f"**Score:** `{entry['summary_score']:.4f}` | **Consistent:** {'✅' if entry['is_consistent'] else '❌'} | **Threshold:** `{entry['threshold']}`")

                        st.markdown("**NLI Pair Matrix** *(rows = doc sentences, cols = summary sentences)*")
                        matrix_df = pd.DataFrame(
                            entry["nli_matrix"],
                            index=[f"D{i+1}: {s[:40]}..." for i, s in enumerate(entry["doc_sentences"])],
                            columns=[f"S{j+1}" for j in range(len(entry["summary_sentences"]))],
                        )
                        st.dataframe(matrix_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1), use_container_width=True)

                        st.markdown("**Per summary sentence scores:**")
                        for j, sent_sc in enumerate(entry["sentence_scores"]):
                            color = "🟢" if sent_sc["final_score"] >= 0.5 else "🔴"
                            with st.expander(f"S{j+1}: {sent_sc['text'][:80]}  {color} score={sent_sc['final_score']:.4f}", expanded=False):
                                st.markdown(f"**Summary sentence:** {sent_sc['text']}")
                                st.markdown(f"**Final score (max over doc sentences):** `{sent_sc['final_score']:.4f}`")
                                doc_scores_df = pd.DataFrame({
                                    "Doc sentence": [f"D{i+1}: {s[:60]}..." for i, s in enumerate(entry["doc_sentences"])],
                                    "Entailment score": sent_sc["raw_scores"],
                                })
                                st.dataframe(doc_scores_df.style.background_gradient(subset=["Entailment score"], cmap="RdYlGn", vmin=0, vmax=1), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("📁 Upload a CSV file to start SummaC evaluation")

# ===== G-EVAL TAB =====
with tab_geval:
    st.markdown("---")
    st.subheader("🤖 G-EVAL — LLM-as-Judge Factual Consistency")
    st.markdown(
        "Scores each (source, summary) pair on a **0–1 scale** using GPT-4 "
        "probability-weighted chain-of-thought. Works natively with Hebrew — no translation required."
    )

    with st.expander("ℹ️ How G-EVAL works", expanded=False):
        st.markdown("""
**Paper:** Liu et al. 2023 — *G-Eval: NLG Evaluation using GPT-4* (arXiv 2303.16634)

**Implementation:** Uses [deepeval](https://docs.confident-ai.com/docs/metrics-llm-evals)'s `GEval` metric,
which implements true probability-weighted scoring via logprobs:

`score = Σ p(token_i) × i  for i ∈ {1..5}` → normalised to **0.0–1.0**

This is more accurate than parsing a text integer — the score reflects the model's full probability
distribution over all 5 score tokens, not just the argmax.

**Score scale:** 0.0 (fully inconsistent) → 1.0 (fully faithful)
- **≥ 0.7** — consistent, claims well supported
- **0.3–0.7** — partial consistency
- **≤ 0.3** — major hallucinations or contradictions
        """)

    geval_csv_file = st.file_uploader(
        "Upload CSV (premise, hypothesis columns required; label optional)",
        type=["csv"],
        key="geval_csv",
    )

    if geval_csv_file is not None:
        try:
            geval_dataframe = pd.read_csv(geval_csv_file)
            st.success(f"✅ Loaded {len(geval_dataframe)} records")

            if not {'premise', 'hypothesis'}.issubset(geval_dataframe.columns):
                st.error("❌ CSV must contain 'premise' and 'hypothesis' columns")
            else:
                if 'label' in geval_dataframe.columns:
                    available_labels = geval_dataframe['label'].dropna().unique().tolist()
                    selected_labels = st.multiselect(
                        "Filter by label (optional):",
                        options=available_labels,
                        default=[],
                        key="ge_label_filter",
                    )
                    if selected_labels:
                        geval_dataframe = geval_dataframe[geval_dataframe['label'].isin(selected_labels)].reset_index(drop=True)
                        st.info(f"📊 Filtered to {len(geval_dataframe)} records")

                col_start, col_end = st.columns(2)
                with col_start:
                    start_index = int(st.number_input("Start index:", min_value=0, max_value=len(geval_dataframe) - 1, value=0, step=1, key="ge_start"))
                with col_end:
                    end_index = int(st.number_input("End index:", min_value=start_index + 1, max_value=len(geval_dataframe), value=min(start_index + 5, len(geval_dataframe)), step=1, key="ge_end"))

                st.markdown("---")
                st.subheader("⚙️ Configuration")

                selected_model = st.selectbox(
                    "OpenAI model:",
                    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4"],
                    key="ge_model",
                    help="gpt-4o-mini is fastest and cheapest; gpt-4o gives highest accuracy.",
                )
                num_parallel_workers = int(st.number_input("Parallel workers:", min_value=1, max_value=32, value=2, step=1, key="ge_workers"))

                if 'geval_stop_event' not in st.session_state:
                    st.session_state['geval_stop_event'] = threading.Event()

                ge_run_col, ge_stop_col = st.columns([3, 1])
                with ge_run_col:
                    ge_run_clicked = st.button(
                        "▶️ Run G-EVAL",
                        type="primary",
                        use_container_width=True,
                        disabled=st.session_state.get('geval_running', False),
                        key="ge_run",
                    )
                with ge_stop_col:
                    ge_stop_clicked = st.button(
                        "⏹ Stop",
                        type="secondary",
                        use_container_width=True,
                        disabled=not st.session_state.get('geval_running', False),
                        key="ge_stop",
                    )

                if ge_stop_clicked:
                    ge_stop_ev = st.session_state.get('geval_stop_event')
                    if ge_stop_ev is not None:
                        ge_stop_ev.set()
                    st.session_state['geval_running'] = False

                if ge_run_clicked:
                    from metrics.llm_as_a_judge.g_eval import GEvalScorer, serialize_result as geval_serialize_result

                    if not os.getenv("OPENAI_API_KEY", ""):
                        st.error("❌ OPENAI_API_KEY not found in .env file.")
                        st.stop()

                    geval_scorer = GEvalScorer(model_name=selected_model)
                    sliced_dataframe = geval_dataframe.iloc[start_index:end_index].copy()
                    total_samples = len(sliced_dataframe)

                    geval_stop_event = threading.Event()
                    st.session_state['geval_stop_event'] = geval_stop_event
                    st.session_state['geval_running'] = True

                    result_log, was_stopped = _run_llm_judge_batch(
                        scorer_fn=geval_scorer.score_one,
                        serialize_fn=geval_serialize_result,
                        sliced_dataframe=sliced_dataframe,
                        start_index=start_index,
                        num_parallel_workers=num_parallel_workers,
                        stop_event=geval_stop_event,
                        progress_label="G-EVAL",
                    )

                    st.session_state['geval_running'] = False

                    if was_stopped:
                        st.warning(f"⏹ Stopped — {sum(r is not None for r in result_log)} / {total_samples} samples completed")
                    else:
                        st.success("✅ Done!")

                    st.session_state['geval_log'] = result_log
                    st.session_state['geval_summary_csv'] = pd.DataFrame([
                        {"index": r["index"], "label": r.get("label", ""), "score": r.get("score"), "error": r.get("error", "")}
                        for r in result_log
                    ]).to_csv(index=False)
                    st.session_state['geval_config_md'] = _build_run_config_md(
                        "G-EVAL",
                        {"name": geval_csv_file.name, "total_size": len(geval_dataframe), "evaluated": total_samples},
                        [
                            ("OpenAI Model", selected_model),
                            ("Parallel Workers", num_parallel_workers),
                            ("Row Range", f"{start_index}–{end_index - 1} ({total_samples} samples)"),
                            ("Temperature", "0.0 (deterministic)"),
                            ("Score Scale", "1–5"),
                        ],
                    )
                    st.success("✅ Done!")

                    summary_dataframe = _build_llm_judge_summary_dataframe(
                        result_log=result_log,
                        sliced_dataframe=sliced_dataframe,
                        core_field_names={
                            "index": "index",
                            "label": "label",
                            "score": "score",
                            "reasoning": "evaluation",
                            "model": "model",
                            "passed": "passed",
                            "error": "error",
                        },
                    )
                    valid_scores = summary_dataframe[summary_dataframe["score"].notna()]["score"]
                    if len(valid_scores):
                        st.metric("Mean G-EVAL Score (0–1)", f"{valid_scores.mean():.4f}")
                    st.dataframe(summary_dataframe, use_container_width=True)

                    if 'label' in sliced_dataframe.columns:
                        labeled_rows = summary_dataframe[
                            summary_dataframe["score"].notna() &
                            summary_dataframe["label"].isin(["entailment", "contradiction"])
                        ]
                        if len(labeled_rows) >= 4:
                            binary_labels = (labeled_rows["label"] == "entailment").astype(int)
                            tau_score, p_value = kendalltau(labeled_rows["score"], binary_labels)
                            st.markdown(f"**Kendall τ vs label:** `{tau_score:.4f}` (p={p_value:.4f})")

                    download_csv_json(
                        csv_bytes=summary_dataframe.to_csv(index=False).encode('utf-8'),
                        json_bytes=json.dumps(result_log, ensure_ascii=False, indent=2).encode('utf-8'),
                        base_filename="geval_results",
                    )

                # ── Trace Viewer ──────────────────────────────────────────────
                if st.session_state.get('geval_log'):
                    trace_log = st.session_state['geval_log']
                    st.markdown("---")
                    if st.session_state.get('geval_config_md') and st.session_state.get('geval_summary_csv'):
                        _zip_buf = io.BytesIO()
                        with zipfile.ZipFile(_zip_buf, 'w', zipfile.ZIP_DEFLATED) as _zf:
                            _zf.writestr("geval_results.csv", st.session_state['geval_summary_csv'])
                            _zf.writestr("geval_config.md",   st.session_state['geval_config_md'])
                        st.download_button("📦 Download Results + Config (ZIP)", _zip_buf.getvalue(), "geval_results.zip", "application/zip", key="dl_zip_geval")
                    st.subheader("🔎 Trace Viewer")
                    selected_sample = st.selectbox(
                        "Select sample:",
                        range(len(trace_log)),
                        format_func=lambda x: f"Sample {trace_log[x]['index']} — score: {trace_log[x].get('score', 'ERR')}",
                        key="ge_trace_select",
                    )
                    trace_entry = trace_log[selected_sample]

                    if trace_entry.get("error"):
                        st.error(f"Pipeline error: {trace_entry['error']}")
                    else:
                        with st.expander("📄 Texts", expanded=False):
                            if trace_entry.get("label"):
                                st.markdown(f"**Label:** `{trace_entry['label']}`")
                            st.markdown("**Hypothesis (summary):**")
                            st.write(trace_entry["hypothesis"])
                            st.markdown("**Premise (source):**")
                            st.write(trace_entry["premise"])

                        score_value = trace_entry.get("score")
                        score_icon = "🟢" if score_value and score_value >= 0.7 else ("🔴" if score_value is not None and score_value <= 0.3 else "🟡")
                        st.markdown(f"**Score:** {score_icon} `{score_value:.4f} / 1.0`" if score_value is not None else "**Score:** —")
                        st.markdown("**Chain-of-thought reasoning:**")
                        st.info(trace_entry.get("evaluation", "—"))

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("📁 Upload a CSV file to start G-EVAL evaluation")
        st.markdown("""
        **Expected CSV format:**
        ```
        premise, hypothesis, label
        source article text..., generated summary..., entailment
        ```
        - `premise` — source document
        - `hypothesis` — generated summary
        - `label` — optional; used for Kendall τ (`entailment` / `contradiction`)
        """)

with tab_vanilla:
    st.markdown("---")
    st.subheader("🗣️ LLM-as-Judge (Vanilla) — Zero-Shot Factual Consistency")
    st.markdown(
        "Scores each (source, summary) pair on a **1–5 scale** by asking GPT directly. "
        "The model's text reply is parsed for a digit. Simpler than G-EVAL — no logprobs."
    )

    with st.expander("ℹ️ How Vanilla LLM-as-Judge works", expanded=False):
        st.markdown("""
**Approach:** Zero-shot prompt to GPT with a 1–5 scoring rubric.

**Score scale:**
- **1** — Major hallucinations or direct contradictions
- **2** — Several unsupported claims; partially inconsistent
- **3** — Mostly consistent; minor unsupported details
- **4** — Consistent with at most one minor gap
- **5** — Fully faithful; every claim grounded in the source

**Difference vs G-EVAL:** G-EVAL reads the probability distribution over score tokens (logprobs)
and computes a weighted average → continuous 0–1 float. This metric parses the digit the
model wrote as text → integer 1–5. Simpler but less precise.
        """)

    vanilla_csv_file = st.file_uploader(
        "Upload CSV (premise, hypothesis columns required; label optional)",
        type=["csv"],
        key="vanilla_csv",
    )

    if vanilla_csv_file is not None:
        try:
            vanilla_dataframe = pd.read_csv(vanilla_csv_file)
            st.success(f"✅ Loaded {len(vanilla_dataframe)} records")

            if not {'premise', 'hypothesis'}.issubset(vanilla_dataframe.columns):
                st.error("❌ CSV must contain 'premise' and 'hypothesis' columns")
            else:
                if 'label' in vanilla_dataframe.columns:
                    available_labels = vanilla_dataframe['label'].dropna().unique().tolist()
                    selected_labels = st.multiselect(
                        "Filter by label (optional):",
                        options=available_labels,
                        default=[],
                        key="vn_label_filter",
                    )
                    if selected_labels:
                        vanilla_dataframe = vanilla_dataframe[vanilla_dataframe['label'].isin(selected_labels)].reset_index(drop=True)
                        st.info(f"📊 Filtered to {len(vanilla_dataframe)} records")

                col_start, col_end = st.columns(2)
                with col_start:
                    start_index = int(st.number_input("Start index:", min_value=0, max_value=len(vanilla_dataframe) - 1, value=0, step=1, key="vn_start"))
                with col_end:
                    end_index = int(st.number_input("End index:", min_value=start_index + 1, max_value=len(vanilla_dataframe), value=min(start_index + 5, len(vanilla_dataframe)), step=1, key="vn_end"))

                st.markdown("---")
                st.subheader("⚙️ Configuration")

                selected_model = st.selectbox(
                    "OpenAI model:",
                    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4"],
                    key="vn_model",
                    help="gpt-4o-mini is fastest and cheapest; gpt-4o gives highest accuracy.",
                )
                num_parallel_workers = int(st.number_input("Parallel workers:", min_value=1, max_value=32, value=2, step=1, key="vn_workers"))

                if 'vanilla_stop_event' not in st.session_state:
                    st.session_state['vanilla_stop_event'] = threading.Event()

                vn_run_col, vn_stop_col = st.columns([3, 1])
                with vn_run_col:
                    vn_run_clicked = st.button(
                        "▶️ Run Vanilla LLM-as-Judge",
                        type="primary",
                        use_container_width=True,
                        disabled=st.session_state.get('vanilla_running', False),
                        key="vn_run",
                    )
                with vn_stop_col:
                    vn_stop_clicked = st.button(
                        "⏹ Stop",
                        type="secondary",
                        use_container_width=True,
                        disabled=not st.session_state.get('vanilla_running', False),
                        key="vn_stop",
                    )

                if vn_stop_clicked:
                    vn_stop_ev = st.session_state.get('vanilla_stop_event')
                    if vn_stop_ev is not None:
                        vn_stop_ev.set()
                    st.session_state['vanilla_running'] = False

                if vn_run_clicked:
                    from metrics.llm_as_a_judge.vanila_zero_shot import VanillaScorer, serialize_result as vanilla_serialize_result
                    from openai import OpenAI as _OpenAI

                    openai_api_key = os.getenv("OPENAI_API_KEY", "")
                    if not openai_api_key:
                        st.error("❌ OPENAI_API_KEY not found in .env file.")
                        st.stop()

                    vanilla_scorer = VanillaScorer(
                        openai_client=_OpenAI(api_key=openai_api_key),
                        model_name=selected_model,
                    )
                    sliced_dataframe = vanilla_dataframe.iloc[start_index:end_index].copy()
                    total_samples = len(sliced_dataframe)

                    vanilla_stop_event = threading.Event()
                    st.session_state['vanilla_stop_event'] = vanilla_stop_event
                    st.session_state['vanilla_running'] = True

                    result_log, was_stopped = _run_llm_judge_batch(
                        scorer_fn=vanilla_scorer.score_one,
                        serialize_fn=vanilla_serialize_result,
                        sliced_dataframe=sliced_dataframe,
                        start_index=start_index,
                        num_parallel_workers=num_parallel_workers,
                        stop_event=vanilla_stop_event,
                        progress_label="Vanilla LLM-as-Judge",
                    )

                    st.session_state['vanilla_running'] = False

                    if was_stopped:
                        st.warning(f"⏹ Stopped — {sum(r is not None for r in result_log)} / {total_samples} samples completed")
                    else:
                        st.success("✅ Done!")

                    st.session_state['vanilla_log'] = result_log
                    st.session_state['vanilla_config_md'] = _build_run_config_md(
                        "Vanilla LLM-as-Judge",
                        {"name": vanilla_csv_file.name, "total_size": len(vanilla_dataframe), "evaluated": total_samples},
                        [
                            ("OpenAI Model", selected_model),
                            ("Parallel Workers", num_parallel_workers),
                            ("Row Range", f"{start_index}–{end_index - 1} ({total_samples} samples)"),
                            ("Temperature", "0.0 (deterministic)"),
                            ("Score Scale", "1–5 integer (text-parsed)"),
                        ],
                    )
                    st.success("✅ Done!")

                    summary_dataframe = _build_llm_judge_summary_dataframe(
                        result_log=result_log,
                        sliced_dataframe=sliced_dataframe,
                        core_field_names={
                            "index": "index",
                            "label": "label",
                            "score (1–5)": "score",
                            "score_normalized (0–1)": "score_normalized",
                            "reasoning": "reasoning",
                            "model": "model",
                            "passed": "passed",
                            "error": "error",
                        },
                    )
                    valid_normalized = summary_dataframe[summary_dataframe["score_normalized (0–1)"].notna()]["score_normalized (0–1)"]
                    if len(valid_normalized):
                        col_raw, col_norm = st.columns(2)
                        with col_raw:
                            raw_mean = summary_dataframe[summary_dataframe["score (1–5)"].notna()]["score (1–5)"].mean()
                            st.metric("Mean Score (1–5)", f"{raw_mean:.3f}")
                        with col_norm:
                            st.metric("Mean Score (0–1)", f"{valid_normalized.mean():.4f}")
                    st.dataframe(summary_dataframe, use_container_width=True)

                    if 'label' in sliced_dataframe.columns:
                        labeled_rows = summary_dataframe[
                            summary_dataframe["score_normalized (0–1)"].notna() &
                            summary_dataframe["label"].isin(["entailment", "contradiction"])
                        ]
                        if len(labeled_rows) >= 4:
                            binary_labels = (labeled_rows["label"] == "entailment").astype(int)
                            tau_score, p_value = kendalltau(labeled_rows["score_normalized (0–1)"], binary_labels)
                            st.markdown(f"**Kendall τ vs label:** `{tau_score:.4f}` (p={p_value:.4f})")

                    download_csv_json(
                        csv_bytes=summary_dataframe.to_csv(index=False).encode('utf-8'),
                        json_bytes=json.dumps(result_log, ensure_ascii=False, indent=2).encode('utf-8'),
                        base_filename="vanilla_results",
                    )

                # ── Trace Viewer ──────────────────────────────────────────────
                if st.session_state.get('vanilla_log'):
                    trace_log = st.session_state['vanilla_log']
                    st.markdown("---")
                    if st.session_state.get('vanilla_config_md'):
                        st.download_button("📋 Download Config (MD)", st.session_state['vanilla_config_md'].encode(), "vanilla_config.md", "text/markdown", key="dl_md_vanilla")
                    st.subheader("🔎 Trace Viewer")
                    selected_sample = st.selectbox(
                        "Select sample:",
                        range(len(trace_log)),
                        format_func=lambda x: f"Sample {trace_log[x]['index']} — score: {trace_log[x].get('score', 'ERR')}",
                        key="vn_trace_select",
                    )
                    trace_entry = trace_log[selected_sample]

                    if trace_entry.get("error"):
                        st.error(f"Pipeline error: {trace_entry['error']}")
                    else:
                        with st.expander("📄 Texts", expanded=False):
                            if trace_entry.get("label"):
                                st.markdown(f"**Label:** `{trace_entry['label']}`")
                            st.markdown("**Hypothesis (summary):**")
                            st.write(trace_entry["hypothesis"])
                            st.markdown("**Premise (source):**")
                            st.write(trace_entry["premise"])

                        raw_score = trace_entry.get("score")
                        score_icon = "🟢" if raw_score and raw_score >= 4 else ("🔴" if raw_score and raw_score <= 2 else "🟡")
                        col_raw, col_norm = st.columns(2)
                        with col_raw:
                            st.markdown(f"**Score:** {score_icon} `{raw_score} / 5`" if raw_score is not None else "**Score:** —")
                        with col_norm:
                            norm = trace_entry.get("score_normalized")
                            st.markdown(f"**Normalized:** `{norm:.4f} / 1.0`" if norm is not None else "**Normalized:** —")
                        st.markdown("**Reasoning:**")
                        st.info(trace_entry.get("reasoning", "—"))

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("📁 Upload a CSV file to start Vanilla LLM-as-Judge evaluation")
        st.markdown("""
        **Expected CSV format:**
        ```
        premise, hypothesis, label
        source article text..., generated summary..., entailment
        ```
        - `premise` — source document
        - `hypothesis` — generated summary
        - `label` — optional; used for Kendall τ (`entailment` / `contradiction`)
        """)

with tab_bartscore:
    st.markdown("---")
    st.subheader("📊 BARTScore — Generation Log-Probability Factual Consistency")
    st.markdown(
        "Scores each (source, summary) pair by force-decoding the summary through BART "
        "and averaging the log-probability of each token. No API key required — runs locally."
    )

    with st.expander("ℹ️ How BARTScore works", expanded=False):
        st.markdown("""
**Paper:** Yuan et al. 2021 — *BARTScore: Evaluating Generated Text as Text Generation* (NeurIPS 2021)

**Mechanism:** BART is given the source document as input and asked to generate the summary token by token.
The score is the average log-probability assigned to each summary token:

`raw_score = (1/N) × Σ log P(token_i | source, token_1 … token_{i-1})`

If a token in the summary is **not supported** by the source, BART assigns it near-zero probability,
which pulls the average down. Faithful summaries score closer to 0; hallucinated ones score much lower (e.g. −4).

**normalized_score = exp(raw_score)** maps this to 0–1 for display.

**TRUE benchmark result:** avg ROC-AUC 72.2 across 11 factuality datasets.
        """)

    bartscore_csv_file = st.file_uploader(
        "Upload CSV (premise, hypothesis columns required; label optional)",
        type=["csv"],
        key="bartscore_csv",
    )

    if bartscore_csv_file is not None:
        try:
            bartscore_dataframe = pd.read_csv(bartscore_csv_file)
            st.success(f"✅ Loaded {len(bartscore_dataframe)} records")

            if not {'premise', 'hypothesis'}.issubset(bartscore_dataframe.columns):
                st.error("❌ CSV must contain 'premise' and 'hypothesis' columns")
            else:
                if 'label' in bartscore_dataframe.columns:
                    available_labels = bartscore_dataframe['label'].dropna().unique().tolist()
                    selected_labels = st.multiselect(
                        "Filter by label (optional):",
                        options=available_labels,
                        default=[],
                        key="bs_label_filter",
                    )
                    if selected_labels:
                        bartscore_dataframe = bartscore_dataframe[bartscore_dataframe['label'].isin(selected_labels)].reset_index(drop=True)
                        st.info(f"📊 Filtered to {len(bartscore_dataframe)} records")

                col_start, col_end = st.columns(2)
                with col_start:
                    start_index = int(st.number_input("Start index:", min_value=0, max_value=len(bartscore_dataframe) - 1, value=0, step=1, key="bs_start"))
                with col_end:
                    end_index = int(st.number_input("End index:", min_value=start_index + 1, max_value=len(bartscore_dataframe), value=min(start_index + 20, len(bartscore_dataframe)), step=1, key="bs_end"))

                st.markdown("---")
                st.subheader("⚙️ Configuration")

                selected_model = st.selectbox(
                    "BART model:",
                    ["facebook/bart-large-cnn", "facebook/bart-large", "facebook/bart-large-mnli"],
                    key="bs_model",
                    help="bart-large-cnn is fine-tuned on CNN/DM summarization — recommended for news.",
                )
                col_batch, col_device = st.columns(2)
                with col_batch:
                    selected_batch_size = int(st.number_input(
                        "Batch size:", min_value=1, max_value=16, value=4, step=1, key="bs_batch",
                        help="Pairs per GPU forward pass. Reduce to 1–2 if you get out-of-memory errors.",
                    ))
                with col_device:
                    import torch as _torch
                    detected_device = "cuda" if _torch.cuda.is_available() else "cpu"
                    st.info(f"🖥️ Device: `{detected_device}` (auto-detected)")

                passing_threshold = st.slider(
                    "Passing threshold (normalized score):",
                    min_value=0.0, max_value=0.5, value=0.1, step=0.01,
                    key="bs_threshold",
                    help="exp(raw_score). Typical faithful summaries score 0.05–0.20.",
                )

                bs_inference_type = st.radio(
                    "Inference type:",
                    ["Local", "Remote API"],
                    horizontal=True,
                    key="bs_inference_type",
                    help="Local: loads BART on this machine (~1.5GB). Remote API: calls inference server on GPU.",
                )
                if bs_inference_type == "Remote API":
                    bs_api_url = st.text_input("API URL", value="http://localhost:8080", key="bs_api_url")

                if st.button("▶️ Run BARTScore", type="primary", use_container_width=True):
                    import requests as _requests
                    sliced_dataframe = bartscore_dataframe.iloc[start_index:end_index].copy()
                    total_samples = len(sliced_dataframe)
                    progress_bar = st.progress(0, text="Starting BARTScore…")
                    result_log = []

                    if bs_inference_type == "Remote API":
                        for i, (_, csv_row) in enumerate(sliced_dataframe.iterrows()):
                            try:
                                resp = _requests.post(
                                    f"{bs_api_url}/bartscore/predict",
                                    json={
                                        "source_document": str(csv_row['premise']),
                                        "summary": str(csv_row['hypothesis']),
                                    },
                                    timeout=60,
                                )
                                if resp.status_code == 200:
                                    r = resp.json()
                                    result_log.append({
                                        "index": start_index + i,
                                        "premise": str(csv_row['premise']),
                                        "hypothesis": str(csv_row['hypothesis']),
                                        "label": str(csv_row.get('label', '')),
                                        "raw_score": r["raw_score"],
                                        "normalized_score": r["normalized_score"],
                                    })
                                else:
                                    result_log.append({"index": start_index + i, "premise": str(csv_row['premise']), "hypothesis": str(csv_row['hypothesis']), "label": str(csv_row.get('label', '')), "raw_score": None, "normalized_score": None, "error": resp.text})
                            except Exception as _api_err:
                                result_log.append({"index": start_index + i, "premise": str(csv_row['premise']), "hypothesis": str(csv_row['hypothesis']), "label": str(csv_row.get('label', '')), "raw_score": None, "normalized_score": None, "error": str(_api_err)})
                            progress_bar.progress((i + 1) / total_samples, text=f"Scored {i + 1} / {total_samples}")
                    else:
                        from dataclasses import asdict as dataclass_asdict
                        from metrics.bartscore import BartScorer, serialize_result as bart_serialize_result

                        document_summary_pairs = list(zip(
                            sliced_dataframe['premise'].astype(str),
                            sliced_dataframe['hypothesis'].astype(str),
                        ))
                        model_load_status = st.empty()
                        model_load_status.info(f"⏳ Loading `{selected_model}`… first run downloads weights (~1.5 GB)")

                        @st.cache_resource
                        def _load_bart_scorer(
                            cached_model_name: str,
                            cached_batch_size: int,
                            cached_threshold: float,
                        ) -> BartScorer:
                            return BartScorer.from_pretrained(
                                model_name=cached_model_name,
                                batch_size=cached_batch_size,
                                passing_threshold=cached_threshold,
                            )

                        bart_scorer = _load_bart_scorer(selected_model, selected_batch_size, passing_threshold)
                        model_load_status.empty()

                        scored_count = 0
                        for chunk_start in range(0, total_samples, selected_batch_size):
                            chunk_end = min(chunk_start + selected_batch_size, total_samples)
                            chunk_pairs = document_summary_pairs[chunk_start:chunk_end]
                            chunk_results = bart_scorer.score_batch(chunk_pairs)
                            for batch_item in chunk_results:
                                csv_row = sliced_dataframe.iloc[batch_item.sample_index]
                                if batch_item.error_message:
                                    result_log.append({"index": start_index + batch_item.sample_index, "premise": str(csv_row['premise']), "hypothesis": str(csv_row['hypothesis']), "label": str(csv_row.get('label', '')), "raw_score": None, "normalized_score": None, "error": batch_item.error_message})
                                else:
                                    result_log.append(dataclass_asdict(bart_serialize_result(
                                        sample_index=start_index + batch_item.sample_index,
                                        source_document=str(csv_row['premise']),
                                        generated_summary=str(csv_row['hypothesis']),
                                        ground_truth_label=str(csv_row.get('label', '')),
                                        eval_result=batch_item.eval_result,
                                    )))
                            scored_count = chunk_end
                            progress_bar.progress(scored_count / total_samples, text=f"Scored {scored_count} / {total_samples}")

                    progress_bar.empty()
                    st.session_state['bartscore_log'] = result_log
                    st.session_state['bartscore_config_md'] = _build_run_config_md(
                        "BARTScore",
                        {"name": bartscore_csv_file.name, "total_size": len(bartscore_dataframe), "evaluated": total_samples},
                        [
                            ("Inference Type", bs_inference_type),
                            ("BART Model", selected_model),
                            ("Batch Size", selected_batch_size if bs_inference_type == "Local" else "N/A (API)"),
                            ("Device", detected_device if bs_inference_type == "Local" else "Remote GPU"),
                            ("Passing Threshold", passing_threshold),
                            ("Row Range", f"{start_index}–{end_index - 1} ({total_samples} samples)"),
                        ],
                    )
                    st.success("✅ Done!")

                    summary_dataframe = pd.DataFrame([
                        {
                            "index": r["index"],
                            "label": r.get("label", ""),
                            "raw_score": r.get("raw_score"),
                            "normalized_score": r.get("normalized_score"),
                            "error": r.get("error", ""),
                        }
                        for r in result_log
                    ])
                    valid_normalized = summary_dataframe[summary_dataframe["normalized_score"].notna()]["normalized_score"]
                    valid_raw = summary_dataframe[summary_dataframe["raw_score"].notna()]["raw_score"]
                    if len(valid_normalized):
                        col_raw, col_norm = st.columns(2)
                        with col_raw:
                            st.metric("Mean Raw Score (log-prob)", f"{valid_raw.mean():.4f}")
                        with col_norm:
                            st.metric("Mean Normalized Score (0–1)", f"{valid_normalized.mean():.4f}")
                    st.dataframe(summary_dataframe, use_container_width=True)

                    if 'label' in sliced_dataframe.columns:
                        labeled_rows = summary_dataframe[
                            summary_dataframe["normalized_score"].notna() &
                            summary_dataframe["label"].isin(["entailment", "contradiction"])
                        ]
                        if len(labeled_rows) >= 4:
                            binary_labels = (labeled_rows["label"] == "entailment").astype(int)
                            tau_score, p_value = kendalltau(labeled_rows["normalized_score"], binary_labels)
                            st.markdown(f"**Kendall τ vs label:** `{tau_score:.4f}` (p={p_value:.4f})")

                    download_csv_json(
                        csv_bytes=summary_dataframe.to_csv(index=False).encode('utf-8'),
                        json_bytes=json.dumps(result_log, ensure_ascii=False, indent=2).encode('utf-8'),
                        base_filename="bartscore_results",
                    )

                # ── Trace Viewer ──────────────────────────────────────────────
                if st.session_state.get('bartscore_log'):
                    trace_log = st.session_state['bartscore_log']
                    st.markdown("---")
                    if st.session_state.get('bartscore_config_md'):
                        st.download_button("📋 Download Config (MD)", st.session_state['bartscore_config_md'].encode(), "bartscore_config.md", "text/markdown", key="dl_md_bartscore")
                    st.subheader("🔎 Trace Viewer")
                    selected_sample = st.selectbox(
                        "Select sample:",
                        range(len(trace_log)),
                        format_func=lambda x: f"Sample {trace_log[x]['index']} — norm: {trace_log[x].get('normalized_score', 'ERR')}",
                        key="bs_trace_select",
                    )
                    trace_entry = trace_log[selected_sample]

                    if trace_entry.get("error"):
                        st.error(f"Pipeline error: {trace_entry['error']}")
                    else:
                        with st.expander("📄 Texts", expanded=False):
                            if trace_entry.get("label"):
                                st.markdown(f"**Label:** `{trace_entry['label']}`")
                            st.markdown("**Hypothesis (summary):**")
                            st.write(trace_entry["hypothesis"])
                            st.markdown("**Premise (source):**")
                            st.write(trace_entry["premise"])

                        norm_value = trace_entry.get("normalized_score")
                        raw_value = trace_entry.get("raw_score")
                        score_icon = "🟢" if norm_value and norm_value >= 0.15 else ("🔴" if norm_value is not None and norm_value <= 0.05 else "🟡")
                        col_raw, col_norm = st.columns(2)
                        with col_raw:
                            st.markdown(f"**Raw score:** `{raw_value:.4f}`" if raw_value is not None else "**Raw score:** —")
                            st.caption("avg log-prob per token (−∞ to 0)")
                        with col_norm:
                            st.markdown(f"**Normalized:** {score_icon} `{norm_value:.4f} / 1.0`" if norm_value is not None else "**Normalized:** —")
                            st.caption("exp(raw) — higher = more faithful")

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("📁 Upload a CSV file to start BARTScore evaluation")
        st.markdown("""
        **Expected CSV format:**
        ```
        premise, hypothesis, label
        source article text..., generated summary..., entailment
        ```
        - `premise` — source document
        - `hypothesis` — generated summary
        - `label` — optional; used for Kendall τ (`entailment` / `contradiction`)
        """)

# ===== MFACT TAB =====
with tab_mfact:
    st.markdown("---")
    st.subheader("🔬 MFact — Hebrew Factual Consistency (NeoDictaBERT)")
    st.markdown(
        "Evaluates factual consistency using a NeoDictaBERT model fine-tuned on the MFact Hebrew dataset. "
        "Runs inference via the remote API."
    )

    # ── API configuration ──────────────────────────────────────────────────
    mfact_api_url = st.text_input("API URL", value="http://localhost:8080", key="mfact_api_url")
    if st.button("🏥 Check API Health", key="mfact_health_check"):
        import requests as _req
        try:
            resp = _req.get(f"{mfact_api_url}/health", timeout=5)
            if resp.status_code == 200:
                h = resp.json()
                if h.get("mfact_model_loaded"):
                    st.success(f"✅ MFact model loaded on {h.get('mfact_device', 'unknown')}")
                else:
                    st.error("❌ MFact model not loaded on remote API")
            else:
                st.error(f"❌ API returned status {resp.status_code}")
        except Exception as _e:
            st.error(f"❌ Cannot reach API: {_e}")

    st.markdown("---")

    # ── File upload ────────────────────────────────────────────────────────
    mfact_file = st.file_uploader(
        "Upload CSV (premise, hypothesis columns required; label optional)",
        type=["csv"],
        key="mfact_csv",
    )

    if mfact_file is not None:
        try:
            df_mfact = pd.read_csv(mfact_file)
            st.success(f"✅ Loaded {len(df_mfact)} records")

            if not {'premise', 'hypothesis'}.issubset(df_mfact.columns):
                st.error("❌ CSV must contain 'premise' and 'hypothesis' columns")
            else:
                # Label filter
                if 'label' in df_mfact.columns:
                    available_labels = df_mfact['label'].dropna().unique().tolist()
                    selected_labels = st.multiselect(
                        "Filter by label (optional):",
                        options=available_labels,
                        default=[],
                        key="mfact_label_filter",
                    )
                    if selected_labels:
                        df_mfact = df_mfact[df_mfact['label'].isin(selected_labels)].reset_index(drop=True)
                        st.info(f"📊 Filtered to {len(df_mfact)} records")

                with st.expander("🔍 Dataset Preview (first 5 rows)"):
                    st.dataframe(df_mfact.head(), use_container_width=True)

                # Row range
                col_s, col_e = st.columns(2)
                with col_s:
                    mf_start = int(st.number_input("Start index:", 0, len(df_mfact) - 1, 0, 1, key="mfact_start"))
                with col_e:
                    mf_end = int(st.number_input("End index:", mf_start + 1, len(df_mfact), min(mf_start + 100, len(df_mfact)), 1, key="mfact_end"))
                st.info(f"📊 Will evaluate rows {mf_start}–{mf_end - 1} ({mf_end - mf_start} samples)")

                if st.button("▶️ Run MFact", type="primary", use_container_width=True):
                    import requests as _requests
                    df_slice = df_mfact.iloc[mf_start:mf_end].copy()
                    result_log = []
                    progress_bar = st.progress(0, text="Running MFact inference...")

                    for i, (_, row) in enumerate(df_slice.iterrows()):
                        try:
                            resp = _requests.post(
                                f"{mfact_api_url}/mfact/predict",
                                json={"premise": str(row['premise']), "hypothesis": str(row['hypothesis']), "return_attention": False},
                                timeout=30,
                            )
                            if resp.status_code == 200:
                                r = resp.json()
                                result_log.append({
                                    "index": mf_start + i,
                                    "premise": str(row['premise'])[:120],
                                    "hypothesis": str(row['hypothesis']),
                                    "label": str(row.get('label', '')) if 'label' in df_slice.columns else '',
                                    "prediction": r["prediction"],
                                    "confidence": round(r["confidence"], 4),
                                    **{f"prob_{k}": round(v, 4) for k, v in r["probabilities"].items()},
                                })
                            else:
                                result_log.append({"index": mf_start + i, "error": resp.text})
                        except Exception as _e:
                            result_log.append({"index": mf_start + i, "error": str(_e)})
                        progress_bar.progress((i + 1) / len(df_slice), text=f"Scored {i + 1} / {len(df_slice)}")

                    progress_bar.empty()
                    st.session_state['mfact_log'] = result_log
                    st.success("✅ Done!")

                if st.session_state.get('mfact_log'):
                    log = st.session_state['mfact_log']
                    result_df = pd.DataFrame([r for r in log if 'error' not in r])
                    errors = [r for r in log if 'error' in r]

                    if errors:
                        st.warning(f"⚠️ {len(errors)} samples failed")

                    if not result_df.empty:
                        st.markdown("---")
                        st.subheader("📊 Results")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Samples scored", len(result_df))
                        with col2:
                            st.metric("Avg confidence", f"{result_df['confidence'].mean():.4f}")
                        with col3:
                            if 'label' in result_df.columns and result_df['label'].ne('').any():
                                valid = result_df[result_df['label'].ne('')]
                                acc = (valid['prediction'] == valid['label']).mean()
                                st.metric("Accuracy vs label", f"{acc:.2%}")

                        st.markdown("#### Prediction Distribution")
                        pred_counts = result_df['prediction'].value_counts().reset_index()
                        pred_counts.columns = ['Prediction', 'Count']
                        st.dataframe(pred_counts, use_container_width=True)

                        # Kendall Tau vs label
                        if 'label' in result_df.columns and result_df['label'].ne('').any():
                            valid = result_df[result_df['label'].ne('')]
                            label_binary = label_to_binary(valid['label'])
                            valid_idx = [i for i, l in enumerate(label_binary) if l is not None]
                            if len(valid_idx) >= 4:
                                prob_cols = [c for c in result_df.columns if c.startswith('prob_')]
                                if prob_cols:
                                    st.markdown("#### Kendall τ Correlation (probability vs label)")
                                    clean_labels = [label_binary[i] for i in valid_idx]
                                    tau_cols = st.columns(len(prob_cols))
                                    for ci, col in enumerate(prob_cols):
                                        scores = [float(valid.iloc[i][col]) for i in valid_idx]
                                        tau_r = KendallTauCalculator(scores, clean_labels, col).compute()
                                        delta_color = "normal" if tau_r.statistic > 0.3 else ("inverse" if tau_r.statistic < -0.1 else "off")
                                        with tau_cols[ci]:
                                            st.metric(col, f"τ = {tau_r.statistic:.4f}", f"p = {tau_r.p_value:.4f}", delta_color=delta_color, help=tau_r.interpretation)

                        st.markdown("#### Detailed Results")
                        with st.expander("🔍 View All Results", expanded=True):
                            st.dataframe(result_df, use_container_width=True)

                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Download Results (CSV)",
                                result_df.to_csv(index=False).encode('utf-8'),
                                "mfact_results.csv", "text/csv",
                            )
                        with col2:
                            st.download_button(
                                "📥 Download Results (JSON)",
                                json.dumps(log, ensure_ascii=False, indent=2).encode('utf-8'),
                                "mfact_results.json", "application/json",
                            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("📁 Upload a CSV with `premise` and `hypothesis` columns to begin")
        st.markdown("""
        **Expected CSV format:**
        ```
        premise, hypothesis, label
        source article text..., claim to verify..., supports
        ```
        - `premise` — source document / article
        - `hypothesis` — claim to verify
        - `label` — optional; used for accuracy and Kendall τ
        """)
