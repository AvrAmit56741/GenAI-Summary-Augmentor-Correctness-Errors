import sys
import os
import json
import streamlit as st
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
    RougeCalculator, BleuCalculator, ChrfCalculator,
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
tab_single, tab_batch, tab_classifier, tab_hallucination, tab_compare, tab_qafacteval = st.tabs(["Single Evaluation", "Batch Evaluation", "Classifier Predictions", "Hallucination Metrics", "Compare Metrics", "QAFactEval"])




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
                    max_samples = len(df_batch)
                    start_index = int(st.number_input(
                        "Start index:",
                        min_value=0,
                        max_value=max_samples - 1,
                        value=0,
                        step=1,
                        key="batch_start_index",
                        help="First row to evaluate (0-based)"
                    ))
                    end_index = int(st.number_input(
                        "End index:",
                        min_value=start_index + 1,
                        max_value=max_samples,
                        value=min(start_index + 100, max_samples),
                        step=1,
                        key="batch_end_index",
                        help="Last row (exclusive) — like Python slicing"
                    ))
                    sample_size = end_index - start_index
                    st.info(f"📊 Will evaluate rows {start_index}–{end_index - 1} ({sample_size} samples)")
                
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
                    df_eval = df_batch.iloc[start_index:end_index].copy()

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results_list = []

                    for row_number, (idx, row) in enumerate(df_eval.iterrows()):
                        status_text.text(f"Evaluating sample {row_number + 1}/{sample_size}...")
                        progress_bar.progress((row_number + 1) / sample_size)
                        
                        premise = str(row['premise'])
                        hypothesis = str(row['hypothesis'])

                        premise_words = set(premise.strip().split())
                        hypothesis_words = hypothesis.strip().split()
                        source_length = len(premise.strip().split())
                        summary_length = len(hypothesis_words)
                        overlap_count = sum(1 for word in hypothesis_words if word in premise_words)
                        compression_ratio = round(summary_length / source_length, 4) if source_length > 0 else 0.0

                        result = {
                            'index': idx,
                            'source_length': source_length,
                            'summary_length': summary_length,
                            'overlap': overlap_count,
                            'compression_ratio': compression_ratio,
                        }

                        rougeCalculator = RougeCalculator(premise, hypothesis)

                        # Compute ROUGE-1
                        if batch_rouge:
                            rouge1 = rougeCalculator.compute_rouge_n(1)
                            result['rouge1_precision'] = rouge1['precision']
                            result['rouge1_recall'] = rouge1['recall']
                            result['rouge1_f1'] = rouge1['f1']

                        # Compute ROUGE-2
                        if batch_rouge2:
                            rouge2 = rougeCalculator.compute_rouge_n(2)
                            result['rouge2_precision'] = rouge2['precision']
                            result['rouge2_recall'] = rouge2['recall']
                            result['rouge2_f1'] = rouge2['f1']

                        # Compute ROUGE-L
                        if batch_rougel:
                            rougel = rougeCalculator.compute_rouge_l()
                            result['rougel_precision'] = rougel['precision']
                            result['rougel_recall'] = rougel['recall']
                            result['rougel_f1'] = rougel['f1']

                        # Compute BLEU
                        if batch_bleu:
                            bleuResult = BleuCalculator(premise, hypothesis).compute_bleu(4)
                            result['bleu'] = bleuResult['bleu']
                            result['bleu_bp'] = bleuResult['brevity_penalty']
                            for ngram_name, ngram_score in bleuResult['scores'].items():
                                result[ngram_name] = ngram_score

                        # Compute ChrF
                        if batch_chrf:
                            chrfResult = ChrfCalculator(premise, hypothesis).compute_chrf(6)
                            result['chrf'] = chrfResult['chrf']

                        # Compute BERTScore (expensive)
                        if batch_bertscore:
                            bert_result = BertScoreCalculator(premise, hypothesis, batch_bert_model).compute_bert_score()
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

                            candidate_metrics = {
                                'BLEU':          'bleu',
                                'ROUGE-1 F1':    'rouge1_f1',
                                'ROUGE-2 F1':    'rouge2_f1',
                                'ROUGE-L F1':    'rougel_f1',
                                'ChrF':          'chrf',
                                'BERTScore F1':  'bertscore_f1',
                            }

                            kendall_results = []
                            for display_name, column_name in candidate_metrics.items():
                                if column_name not in results_df.columns:
                                    continue
                                scores = [results_df.iloc[i][column_name] for i in valid_indices]
                                result = KendallTauCalculator(scores, label_binary_clean, display_name).compute()
                                kendall_results.append(result)

                            if kendall_results:
                                cols = st.columns(min(3, len(kendall_results)))
                                for idx, result in enumerate(kendall_results):
                                    delta_color = "normal" if result.statistic > 0.3 else ("inverse" if result.statistic < -0.1 else "off")
                                    with cols[idx % 3]:
                                        st.metric(
                                            result.metric_name,
                                            f"τ = {result.statistic:.4f}",
                                            f"p = {result.p_value:.4f}",
                                            delta_color=delta_color,
                                            help=result.interpretation
                                        )

                                with st.expander("📖 Understanding Kendall Tau Results"):
                                    st.markdown(KENDALL_TAU_EXPLANATION)
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
            import pandas as pd
            import numpy as np

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
        import pandas as pd
        import math as _math

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
                        # Convert labels to binary
                        label_binary = []
                        for label in df_compare['label']:
                            if isinstance(label, str) and label.lower() in ['entailment', 'true', '1']:
                                label_binary.append(1)
                            elif isinstance(label, str) and label.lower() in ['contradiction', 'false', '0']:
                                label_binary.append(0)
                            elif isinstance(label, (int, float)) and not _math.isnan(label):
                                label_binary.append(int(label))
                            else:
                                label_binary.append(None)

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
                            comparison_results.sort(key=lambda r: abs(r.statistic) if not _math.isnan(r.statistic) else -1, reverse=True)

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
        import pandas as pd

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
                    answer_backend = st.radio("Backend:", ["DictaBERT (HF)", "OpenAI"], key="qa_selector")

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

                max_workers = int(st.number_input("Parallel workers:", min_value=1, max_value=8, value=4, step=1, key="qa_workers", help="Number of samples evaluated simultaneously. Higher = faster but more memory/API quota."))

                openai_model = "gpt-4o-mini"
                if needs_openai:
                    st.markdown("---")
                    openai_model = st.selectbox("OpenAI Model:", ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"], key="qa_openai_model")

                if st.button("🔍 Run QAFactEval", type="primary", use_container_width=True):
                    from metrics.qa_facteval import (
                        PipelineConfig, QaFactEvalPipeline,
                        DictaBertAnswerSelector, OpenAiAnswerSelector,
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

                    selector = (
                        DictaBertAnswerSelector()
                        if answer_backend == "DictaBERT (HF)"
                        else OpenAiAnswerSelector(openai_client, openai_model)
                    )
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
                    df_slice = df_qa.iloc[qa_start:qa_end].copy()
                    pairs = [
                        (str(row['hypothesis']), str(row['premise']))
                        for _, row in df_slice.iterrows()
                    ]

                    with st.spinner(f"Running QAFactEval on {len(pairs)} samples ({max_workers} threads)..."):
                        batch = pipeline_obj.run_batch(pairs, max_workers=max_workers)

                    all_qa_results = []
                    for i, item in enumerate(batch):
                        row = df_slice.iloc[i]
                        if item.get('error'):
                            all_qa_results.append({'index': qa_start + i, 'error': item['error'], 'f1_score': 0.0, 'qa_answers': []})
                        else:
                            result = item['result']
                            answerable = sum(1 for a in result.qa_answers if a.is_answerable)
                            all_qa_results.append({
                                'index': qa_start + i,
                                'hypothesis': str(row['hypothesis']),
                                'premise': str(row['premise']),
                                'label': str(row['label']) if 'label' in row.index else '',
                                'questions_generated': result.questions_generated,
                                'questions_after_filter': result.questions_after_filter,
                                'answerable': answerable,
                                'f1_score': round(result.f1_score, 4),
                                'qa_answers': [
                                    {
                                        'np_chunk': a.np_chunk,
                                        'question': a.question,
                                        'summary_answer': a.summary_answer,
                                        'source_answer': a.source_answer,
                                        'is_answerable': a.is_answerable,
                                        'answer_f1': round(a.answer_f1, 4),
                                    }
                                    for a in result.qa_answers
                                ],
                            })

                    st.success("✅ Done!")
                    st.session_state['qafacteval_log'] = all_qa_results

                    summary_rows = [{k: v for k, v in r.items() if k not in ('qa_answers', 'hypothesis', 'premise', 'label')} for r in all_qa_results]
                    result_df = pd.DataFrame(summary_rows)
                    mean_f1 = result_df['f1_score'].mean()

                    st.markdown("---")
                    score_label = {
                        "GPT Judge": "Mean QAFactEval Score (GPT Judge)",
                        "Exact Match": "Mean QAFactEval Score (Exact Match)",
                    }.get(cmp_method, "Mean QAFactEval F1 Score")
                    st.metric(score_label, f"{mean_f1:.4f}")
                    st.dataframe(result_df, use_container_width=True)

                    col_csv, col_json = st.columns(2)
                    with col_csv:
                        st.download_button("📥 Download Summary (CSV)", result_df.to_csv(index=False).encode('utf-8'), "qafacteval_results.csv", "text/csv")
                    with col_json:
                        st.download_button("📥 Download Full Log (JSON)", json.dumps(all_qa_results, ensure_ascii=False, indent=2).encode('utf-8'), "qafacteval_log.json", "application/json")

                if st.session_state.get('qafacteval_log'):
                    log = st.session_state['qafacteval_log']
                    st.markdown("---")
                    summary_rows_dl = [{k: v for k, v in r.items() if k not in ('qa_answers', 'hypothesis', 'premise', 'label')} for r in log]
                    col_csv_dl, col_json_dl = st.columns(2)
                    with col_csv_dl:
                        st.download_button("📥 Download Summary (CSV)", pd.DataFrame(summary_rows_dl).to_csv(index=False).encode('utf-8'), "qafacteval_results.csv", "text/csv", key="dl_csv_persistent")
                    with col_json_dl:
                        st.download_button("📥 Download Full Log (JSON)", json.dumps(log, ensure_ascii=False, indent=2).encode('utf-8'), "qafacteval_log.json", "application/json", key="dl_json_persistent")
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
