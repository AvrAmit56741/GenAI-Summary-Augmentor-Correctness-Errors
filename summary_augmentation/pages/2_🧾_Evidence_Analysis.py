import sys
import os
import json
import streamlit as st
import pandas as pd
import requests

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidence_engine import (
    load_augmentation,
    analyze_claim,
    analyze_summary,
    analyze_all,
    has_mini_lm,
)

st.set_page_config(page_title="Evidence Analysis", layout="wide", page_icon="🧾")

st.title("🧾 Evidence Analysis")
st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📊 Evidence Analysis", "🤖 Model Inference API"])

# ============================================================================
# TAB 1: EVIDENCE ANALYSIS (Original functionality)
# ============================================================================
with tab1:
    st.markdown("Upload augmentation JSON files and extract evidence supporting original claims while contrasting transformed claims.")

    # Upload files
    uploaded_files = st.file_uploader("Upload augmentation JSON file(s)", type=["json"], accept_multiple_files=True, key="evidence_upload")

    # Parameters
    with st.expander("⚙️ Parameters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            top_k = st.number_input("Top-K sentences", min_value=1, max_value=20, value=5, step=1)
        with col2:
            min_score = st.slider("Min similarity threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
        with col3:
            metric_options = ["sequence_matcher", "jaccard", "mini_lm"]
            metric = st.selectbox("Similarity metric", metric_options, index=0)
            if metric == "mini_lm" and not has_mini_lm():
                st.warning("'mini_lm' requires sentence-transformers. Falling back at runtime if unavailable.")
        with col4:
            delta_min = st.slider("Min delta (orig − transformed)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    mode = st.radio("Mode", ["Specific claim", "One summary", "All summaries"], index=0, horizontal=True)

    # Cache parsing
    @st.cache_data(show_spinner=False)
    def _load_all(files):
        docs = []
        for f in files:
            try:
                docs.extend(json.load(f))
            except Exception:
                try:
                    docs.append(json.load(f))
                except Exception:
                    continue
        return docs

    if uploaded_files:
        docs = _load_all(uploaded_files)

        # Build selectors
        doc_id_to_idx = {}
        for i, d in enumerate(docs):
            key = str(d.get("doc_id", f"doc_{i}"))
            doc_id_to_idx[key] = i

        if mode == "Specific claim":
            doc_choice = st.selectbox("Select document", list(doc_id_to_idx.keys()))
            d_idx = doc_id_to_idx[doc_choice]
            summaries = docs[d_idx].get("correctness_corrupted_summaries", [])
            s_idx = st.number_input("Summary index", min_value=0, max_value=max(0, len(summaries)-1), value=0, step=1)

            # Extract claims
            claims = []
            if 0 <= s_idx < len(summaries):
                for t in summaries[s_idx].get("transformation_list", []):
                    if isinstance(t, dict):
                        claims.append(t)
            # Show candidate claims preview
            if claims:
                with st.expander("📋 Candidate Claims in Selected Summary", expanded=False):
                    for idx, t in enumerate(claims):
                        st.markdown(f"**{idx}**. Original: {t.get('original_claim','')}")
                        st.markdown(f"  Transformed: {t.get('transformed_claim','')}")
                        st.markdown("<hr>", unsafe_allow_html=True)

            c_idx = st.number_input("Claim index", min_value=0, max_value=max(0, len(claims)-1), value=0, step=1)

            if st.button("Extract Evidence", type="primary"):
                if 0 <= c_idx < len(claims):
                    t = claims[c_idx]
                    res = analyze_claim(
                        t.get("original_claim", ""),
                        t.get("transformed_claim", ""),
                        docs[d_idx].get("source_article", ""),
                        top_k=top_k,
                        min_score=min_score,
                        metric=metric,
                        delta_min=delta_min,
                    )
                    st.subheader("Result for selected claim")
                    st.json(res)
                    st.download_button("📥 Download JSON", json.dumps(res, ensure_ascii=False, indent=2), "evidence_claim.json", "application/json")

        elif mode == "One summary":
            doc_choice = st.selectbox("Select document", list(doc_id_to_idx.keys()))
            d_idx = doc_id_to_idx[doc_choice]
            summaries = docs[d_idx].get("correctness_corrupted_summaries", [])
            s_idx = st.number_input("Summary index", min_value=0, max_value=max(0, len(summaries)-1), value=0, step=1)

            if st.button("Extract Evidence for Summary", type="primary"):
                res_list = analyze_summary(
                    docs[d_idx],
                    int(s_idx),
                    {"top_k": top_k, "min_score": min_score, "metric": metric, "delta_min": delta_min},
                )
                st.subheader("Results for summary")
                st.json(res_list)
                st.download_button("📥 Download JSON", json.dumps(res_list, ensure_ascii=False, indent=2), "evidence_summary.json", "application/json")

        else:  # All summaries
            if st.button("Extract Evidence for All", type="primary"):
                res_map = analyze_all(
                    docs,
                    {"top_k": top_k, "min_score": min_score, "metric": metric, "delta_min": delta_min},
                )
                st.subheader("Results for all documents")
                st.json(res_map)
                st.download_button("📥 Download JSON", json.dumps(res_map, ensure_ascii=False, indent=2), "evidence_all.json", "application/json")

    else:
        st.info("📁 Upload augmentation JSON file(s) to begin")

# ============================================================================
# TAB 2: MODEL INFERENCE API
# ============================================================================
with tab2:
    st.markdown("### 🤖 Model Inference API")
    st.markdown("Send inference requests to the model API server.")
    
    # API Configuration
    api_url = st.text_input("API URL", value="http://localhost:8081", help="Port 8080=Binary only, Port 8081=Binary+LLM")
    
    # Model selection
    model_type = st.selectbox("Model Type", ["Binary NLI Classifier", "LLM Error Detector"], index=0)
    
    st.markdown("---")
    
    # Input mode selection
    if model_type == "Binary NLI Classifier":
        input_mode = st.radio("Input Mode", ["Manual Input", "CSV Upload"], horizontal=True)
    else:  # LLM
        input_mode = st.radio("Input Mode", ["Manual Input", "CSV Upload", "JSON Upload"], horizontal=True)
    
    # Variables for binary classifier
    premise = ""
    hypothesis = ""
    true_label = None
    
    # Variables for LLM
    article = ""
    summary = ""
    claim = ""
    expected_error = None
    
    if input_mode == "JSON Upload" and model_type == "LLM Error Detector":
        # JSON augmentation file upload for LLM
        uploaded_json = st.file_uploader("Upload augmentation JSON file", type=["json"], key="json_inference")
        
        if uploaded_json is not None:
            try:
                # Load JSON
                json_data = json.load(uploaded_json)
                
                # Handle both single doc and list of docs
                if isinstance(json_data, list):
                    docs = json_data
                else:
                    docs = [json_data]
                
                st.success(f"✅ Loaded {len(docs)} document(s)")
                
                # Document selection
                doc_options = [f"Doc {i}: {d.get('doc_id', f'doc_{i}')}" for i, d in enumerate(docs)]
                selected_doc_str = st.selectbox("Select Document", doc_options)
                doc_idx = int(selected_doc_str.split(":")[0].replace("Doc ", ""))
                
                selected_doc = docs[doc_idx]
                summaries = selected_doc.get("correctness_corrupted_summaries", [])
                
                # Summary selection
                summary_idx = st.number_input("Summary Index", min_value=0, max_value=max(0, len(summaries)-1), value=0, step=1)
                
                if summary_idx < len(summaries):
                    selected_summary = summaries[summary_idx]
                    transformations = selected_summary.get("transformation_list", [])
                    
                    # Claim selection - extract error types from applied_transformations
                    claim_options = []
                    for i, t in enumerate(transformations):
                        # Get error types from applied_transformations list
                        applied = t.get("applied_transformations", [])
                        error_types = [a.get("type", "unknown") for a in applied] if applied else ["unknown"]
                        error_str = ", ".join(error_types[:2])  # Show first 2 error types
                        if len(error_types) > 2:
                            error_str += f" +{len(error_types)-2} more"
                        claim_options.append(f"Claim {i}: {error_str}")
                    
                    if claim_options:
                        selected_claim_str = st.selectbox("Select Claim", claim_options)
                        claim_idx = int(selected_claim_str.split(":")[0].replace("Claim ", ""))
                        
                        if claim_idx < len(transformations):
                            selected_transform = transformations[claim_idx]
                            
                            # Extract data
                            article = selected_doc.get("source_article", "")
                            # Try different summary field names
                            summary = selected_summary.get("original_summary") or selected_summary.get("corrupted_summary") or selected_doc.get("gold_summary", "")
                            claim = selected_transform.get("transformed_claim", "")
                            
                            # Extract error types from applied_transformations
                            applied = selected_transform.get("applied_transformations", [])
                            expected_error = ", ".join([a.get("type", "") for a in applied]) if applied else None
                            
                            # Display selection
                            st.markdown("---")
                            st.markdown("### Selected Sample:")
                            st.markdown(f"**Doc ID**: {selected_doc.get('doc_id', 'N/A')}")
                            st.markdown(f"**Summary Index**: {summary_idx}")
                            st.markdown(f"**Claim Index**: {claim_idx}")
                            if expected_error:
                                st.markdown(f"**Expected Error Type**: {expected_error}")
                            st.markdown(f"**Article**: {article[:200]}..." if len(article) > 200 else f"**Article**: {article}")
                            st.markdown(f"**Summary**: {summary[:200]}..." if len(summary) > 200 else f"**Summary**: {summary}")
                            st.markdown(f"**Claim**: {claim}")
                            
                            # Show original claim for comparison
                            if selected_transform.get("original_claim"):
                                with st.expander("📋 Original vs Transformed"):
                                    st.markdown(f"**Original**: {selected_transform['original_claim']}")
                                    st.markdown(f"**Transformed**: {claim}")
                    else:
                        st.warning("No claims found in selected summary")
                
            except Exception as e:
                st.error(f"❌ Error loading JSON: {str(e)}")
    
    elif input_mode == "CSV Upload":
        # CSV file upload
        if model_type == "Binary NLI Classifier":
            uploaded_csv = st.file_uploader("Upload CSV file (columns: premise, hypothesis, label, ...)", type=["csv"], key="csv_inference")
            required_cols = ["premise", "hypothesis"]
        else:  # LLM
            uploaded_csv = st.file_uploader("Upload CSV file (columns: article, summary, claim, error_type, ...)", type=["csv"], key="csv_inference")
            required_cols = ["article", "summary", "claim"]
        
        if uploaded_csv is not None:
            try:
                # Load CSV
                df = pd.read_csv(uploaded_csv)
                
                # Validate required columns
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    st.success(f"✅ Loaded {len(df)} samples")
                    
                    # Display column info
                    st.markdown(f"**Columns**: {', '.join(df.columns.tolist())}")
                    
                    # Sample selection
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        sample_idx = st.number_input("Select Sample Index", min_value=0, max_value=len(df)-1, value=0, step=1)
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        load_sample = st.button("📥 Load Sample", type="secondary")
                    
                    # Show preview of all samples
                    with st.expander("📋 Preview CSV Data"):
                        st.dataframe(df, use_container_width=True)
                    
                    # Load selected sample
                    if load_sample or sample_idx is not None:
                        selected_row = df.iloc[sample_idx]
                        
                        if model_type == "Binary NLI Classifier":
                            premise = str(selected_row['premise'])
                            hypothesis = str(selected_row['hypothesis'])
                            if 'label' in df.columns:
                                true_label = selected_row['label']
                            
                            st.markdown("---")
                            st.markdown("### Selected Sample:")
                            st.markdown(f"**Index**: {sample_idx}")
                            if true_label is not None:
                                st.markdown(f"**True Label**: {true_label}")
                            st.markdown(f"**Premise**: {premise[:200]}..." if len(premise) > 200 else f"**Premise**: {premise}")
                            st.markdown(f"**Hypothesis**: {hypothesis[:200]}..." if len(hypothesis) > 200 else f"**Hypothesis**: {hypothesis}")
                        
                        else:  # LLM
                            article = str(selected_row['article'])
                            summary = str(selected_row['summary'])
                            claim = str(selected_row['claim'])
                            if 'error_type' in df.columns:
                                expected_error = selected_row['error_type']
                            
                            st.markdown("---")
                            st.markdown("### Selected Sample:")
                            st.markdown(f"**Index**: {sample_idx}")
                            if expected_error is not None:
                                st.markdown(f"**Expected Error**: {expected_error}")
                            st.markdown(f"**Article**: {article[:200]}..." if len(article) > 200 else f"**Article**: {article}")
                            st.markdown(f"**Summary**: {summary[:200]}..." if len(summary) > 200 else f"**Summary**: {summary}")
                            st.markdown(f"**Claim**: {claim[:200]}..." if len(claim) > 200 else f"**Claim**: {claim}")
                        
            except Exception as e:
                st.error(f"❌ Error loading CSV: {str(e)}")
    
    else:  # Manual Input
        # Input fields based on model type
        if model_type == "Binary NLI Classifier":
            premise = st.text_area("Premise", height=150, placeholder="Enter the premise text here...", value=premise)
            hypothesis = st.text_area("Hypothesis", height=150, placeholder="Enter the hypothesis text here...", value=hypothesis)
        else:  # LLM
            article = st.text_area("Article", height=150, placeholder="Enter the source article text here...", value=article)
            summary = st.text_area("Summary", height=100, placeholder="Enter the summary text here...", value=summary)
            claim = st.text_area("Claim", height=100, placeholder="Enter the claim to verify...", value=claim)
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        if model_type == "Binary NLI Classifier":
            return_attention = st.checkbox("Return Attention Weights", value=False)
    with col2:
        if model_type == "LLM Error Detector":
            prompt_format = st.selectbox("Prompt Format", ["alpaca", "gemma", "chatml"], index=0)
    
    # Inference button
    if st.button("🚀 Run Inference", type="primary"):
        # Validation based on model type
        if model_type == "Binary NLI Classifier":
            if not premise or not hypothesis:
                st.error("Please enter both premise and hypothesis")
                st.stop()
        else:  # LLM
            if not article or not summary or not claim:
                st.error("Please enter article, summary, and claim")
                st.stop()
        
        try:
            with st.spinner("Running inference..."):
                # Call appropriate API endpoint
                if model_type == "Binary NLI Classifier":
                    response = requests.post(
                        f"{api_url}/binary/predict",
                        json={
                            "premise": premise,
                            "hypothesis": hypothesis,
                            "return_attention": return_attention
                        },
                        timeout=30
                    )
                else:  # LLM
                    response = requests.post(
                        f"{api_url}/llm/predict",
                        json={
                            "article": article,
                            "summary": summary,
                            "claim": claim,
                            "prompt_format": prompt_format
                        },
                        timeout=180  # Unsloth needs more time (first inference 60-120s)
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Inference completed!")
                    
                    # Display results based on model type
                    if model_type == "Binary NLI Classifier":
                        # Prediction with true label comparison
                        st.markdown(f"### Prediction: **{result['prediction']}**")
                        st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                        
                        # If we have true label from CSV, show comparison
                        if true_label is not None:
                            pred_normalized = str(result['prediction']).lower().strip()
                            true_normalized = str(true_label).lower().strip()
                            is_correct = (pred_normalized == true_normalized) or \
                                        (pred_normalized in ['label_1', 'entailment', '1'] and true_normalized in ['label_1', 'entailment', '1']) or \
                                        (pred_normalized in ['label_0', 'contradiction', '0'] and true_normalized in ['label_0', 'contradiction', '0'])
                            
                            if is_correct:
                                st.success(f"✅ **Correct!** True label: {true_label}")
                            else:
                                st.error(f"❌ **Incorrect!** True label: {true_label}")
                        
                        # Probabilities
                        st.markdown("#### Probabilities:")
                        prob_df = pd.DataFrame([result['probabilities']]).T
                        prob_df.columns = ['Probability']
                        st.dataframe(prob_df, use_container_width=True)
                        
                        # Attention weights (if returned)
                        if return_attention and result.get('attention_top_tokens'):
                            st.markdown("#### Top Attention Tokens:")
                            attn_data = [(t['token'], t['weight']) for t in result['attention_top_tokens']]
                            attn_df = pd.DataFrame(attn_data, columns=['Token', 'Weight'])
                            st.dataframe(attn_df, use_container_width=True)
                    
                    else:  # LLM
                        # Display generated text
                        st.markdown("### Generated Response:")
                        st.markdown(result['generated_text'])
                        
                        # Display parsed JSON if available
                        if result.get('parsed_json'):
                            st.markdown("#### Parsed Error Detection:")
                            parsed = result['parsed_json']
                            
                            # Show error type
                            if 'error_type' in parsed:
                                st.markdown(f"**Error Type**: {parsed['error_type']}")
                                
                                # Compare with expected if available
                                if expected_error is not None:
                                    if str(parsed['error_type']).lower() == str(expected_error).lower():
                                        st.success(f"✅ **Correct!** Expected: {expected_error}")
                                    else:
                                        st.error(f"❌ **Incorrect!** Expected: {expected_error}")
                            
                            # Show full parsed JSON
                            st.json(parsed)
                        else:
                            st.warning("⚠️ Could not parse JSON from model output")
                    
                    # Full JSON response
                    with st.expander("📄 Full JSON Response"):
                        st.json(result)
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API server. Make sure it's running and the URL is correct.")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The model might be loading or processing is taking too long.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # API Health Check
    st.markdown("---")
    if st.button("🏥 Check API Health"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                binary_loaded = health.get('binary_model_loaded', health.get('model_loaded', False))
                llm_loaded = health.get('llm_model_loaded', False)
                
                st.markdown(f"**Device**: {health.get('device', 'unknown')}")
                
                if binary_loaded:
                    st.success("✅ Binary NLI Classifier loaded")
                else:
                    st.warning("⚠️ Binary NLI Classifier not loaded")
                
                if llm_loaded:
                    st.success("✅ LLM Error Detector loaded")
                else:
                    st.warning("⚠️ LLM Error Detector not loaded")
            else:
                st.error(f"❌ API returned status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot reach API: {str(e)}")
