import sys
import os
import json
import streamlit as st
import pandas as pd

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
                    st.markdown(f"  Transformed: {t.get('transformed_claim','')}")
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