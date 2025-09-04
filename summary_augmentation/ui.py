# import streamlit as st
# import json
# import pandas as pd
# from difflib import SequenceMatcher
# import io
# from sklearn.model_selection import train_test_split, StratifiedKFold
#
# st.set_page_config(page_title="Summary & CSV Analyzer & Utils", layout="wide")
#
# # Helper to highlight word‐level diffs
# def highlight_diff(old: str, new: str) -> str:
#     old_words = old.split()
#     new_words = new.split()
#     matcher = SequenceMatcher(None, old_words, new_words)
#     result = []
#     for tag, i1, i2, j1, j2 in matcher.get_opcodes():
#         if tag == "equal":
#             result.append(" ".join(old_words[i1:i2]))
#         elif tag == "delete":
#             result.append(
#                 f"<span style='background-color:salmon;text-decoration:line-through;'>{' '.join(old_words[i1:i2])}</span>"
#             )
#         elif tag == "insert":
#             result.append(
#                 f"<span style='background-color:lightgreen;'>{' '.join(new_words[j1:j2])}</span>"
#             )
#         elif tag == "replace":
#             result.append(
#                 f"<span style='background-color:salmon;text-decoration:line-through;'>{' '.join(old_words[i1:i2])}</span>"
#             )
#             result.append(
#                 f"<span style='background-color:lightgreen;'>{' '.join(new_words[j1:j2])}</span>"
#             )
#     return " ".join(result)
#
# # Create tabs
# tab_json, tab_csv, tab_utils = st.tabs(["JSON Analyzer", "CSV Reviewer", "Utils"])
#
# ### JSON Analyzer ###
# with tab_json:
#     st.title("JSON Summary Analyzer")
#     uploaded_file = st.file_uploader("Upload JSON file", type=["json"], key="uploader")
#     if uploaded_file:
#         data = json.load(uploaded_file)
#         st.sidebar.header("Navigation")
#         idx = st.sidebar.number_input("Index", 0, len(data)-1, 0, 1, key="entry_idx")
#         entry = data[idx]
#
#         st.header("Source Article")
#         st.text_area(
#             "",
#             entry.get("source_article", ""),
#             height=200,
#             key=f"src{idx}",
#             label_visibility="hidden",
#         )
#
#         st.header("Gold Summary")
#         gold = entry.get("gold_summary", "")
#         st.text_area(
#             "",
#             gold,
#             height=150,
#             key=f"gold{idx}",
#             label_visibility="hidden",
#         )
#
#         corrupted = entry.get("correctness_corrupted_summaries", [])
#         if corrupted:
#             sel = st.selectbox("Corrupted #", range(len(corrupted)), key="cidx")
#             corr = corrupted[sel]
#
#             st.subheader("Error Types")
#             # use the actual field if it exists, else fall back to transformation_list
#             if "error_types" in corr:
#                 errs = corr.get("error_types", [])
#             else:
#                 errs = {
#                     t["type"]
#                     for change in corr.get("transformation_list", [])
#                     for t in change.get("applied_transformations", [])
#                 }
#
#             if errs:
#                 st.write(", ".join(sorted(errs)))
#             else:
#                 st.write("No error types found.")
#
#             st.subheader("Diff")
#             st.markdown(
#                 highlight_diff(gold, corr.get("corrupted_summary", "")),
#                 unsafe_allow_html=True
#             )
#
#
# ### CSV Reviewer ###
# with tab_csv:
#     st.title("CSV Reviewer")
#     csv_file = st.file_uploader("Upload CSV (premise, hypothesis, label[, error_types])", type=["csv"], key="csv_uploader")
#     if csv_file:
#         df = pd.read_csv(csv_file)
#         has_err = 'error_types' in df.columns
#         st.subheader("Data Preview")
#         st.dataframe(df, use_container_width=True)
#
#         # Custom comparison
#         st.subheader("Custom Text Comparison")
#         a = st.text_area("Text A", key="ta")
#         b = st.text_area("Text B", key="tb")
#         if a and b:
#             st.markdown(highlight_diff(a, b), unsafe_allow_html=True)
#
#         # Inspect a row (RTL)
#         st.subheader("Inspect a Row")
#         ridx = st.number_input("בחר שורה", 0, len(df)-1, 0, 1, key="ridx")
#         row = df.iloc[ridx]
#         st.markdown(f"<div dir='rtl'>Premise: {row['premise']}</div>", unsafe_allow_html=True)
#         st.markdown(f"<div dir='rtl'>Hypothesis: {row['hypothesis']}</div>", unsafe_allow_html=True)
#         st.markdown(f"<div dir='rtl'>Label: {row['label']}</div>", unsafe_allow_html=True)
#         if has_err:
#             st.markdown(f"<div dir='rtl'>Error Types: {row['error_types']}</div>", unsafe_allow_html=True)
#
#         # Label & error stats
#         st.subheader("Label Distribution")
#         st.bar_chart(df['label'].value_counts())
#         if has_err:
#             st.subheader("Error Types Distribution")
#             exploded = df['error_types'].dropna().str.split(',', expand=True).stack().str.strip()
#             st.bar_chart(exploded.value_counts())
#
#         # Word length histogram
#         st.subheader("Word Length Histogram")
#         lens = pd.DataFrame({
#             'Premise': df['premise'].str.split().str.len(),
#             'Hypothesis': df['hypothesis'].str.split().str.len()
#         })
#         pc = lens['Premise'].value_counts().sort_index()
#         hc = lens['Hypothesis'].value_counts().sort_index()
#         counts_df = pd.DataFrame({'Premise': pc, 'Hypothesis': hc}).fillna(0)
#         st.bar_chart(counts_df)
#
# ### Utils ###
# with tab_utils:
#     st.title("Utils")
#     # Combine CSV
#     st.subheader("Combine CSV Files")
#     csvs = st.file_uploader("CSV to combine", type=["csv"], accept_multiple_files=True, key="ccomb")
#     if csvs:
#         dfc = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
#         st.download_button("Download CSV", dfc.to_csv(index=False).encode('utf-8'), "combined.csv")
#     # Combine JSON
#     st.subheader("Combine JSON Files")
#     jsns = st.file_uploader("JSON to combine", type=["json"], accept_multiple_files=True, key="jcomb")
#     if jsns:
#         lst = []
#         for f in jsns:
#             lst.extend(json.load(f))
#         st.download_button("Download JSON", json.dumps(lst, ensure_ascii=False), "combined.json")
#     # Split Data
#     st.subheader("Split Data")
#     split_mode = st.selectbox("Mode", ["kfold", "naive"], key="mode")
#     if csv_file:
#         df2 = df.copy()
#         if split_mode == 'kfold':
#             k = st.number_input("K folds", 2, 20, 5, key="k")
#             skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
#             for i, (train_idx, test_idx) in enumerate(skf.split(df2, df2['label'])):
#                 tr = df2.iloc[train_idx]
#                 te = df2.iloc[test_idx]
#                 st.download_button(f"Download train_{i}", tr.to_csv(index=False).encode(), f"train_{i}.csv")
#                 st.download_button(f"Download test_{i}", te.to_csv(index=False).encode(), f"test_{i}.csv")
#         else:
#             pval = st.slider("Validation %", 0.0, 0.5, 0.1, key="pval")
#             ptr = st.slider("Test %", 0.0, 0.5, 0.1, key="ptr")
#             train, temp = train_test_split(df2, test_size=(pval+ptr), stratify=df2['label'], random_state=42)
#             val, test = train_test_split(temp, test_size=ptr/(pval+ptr), stratify=temp['label'], random_state=42)
#             st.download_button("Download train", train.to_csv(index=False).encode(), "train.csv")
#             st.download_button("Download val", val.to_csv(index=False).encode(), "val.csv")
#             st.download_button("Download test", test.to_csv(index=False).encode(), "test.csv")
import streamlit as st
import json
import pandas as pd
from difflib import SequenceMatcher
import re
from sklearn.model_selection import train_test_split, StratifiedKFold

st.set_page_config(page_title="Summary & CSV Analyzer & Utils", layout="wide")

# Helper to highlight word‐level diffs
def highlight_diff(old: str, new: str) -> str:
    old_words = old.split()
    new_words = new.split()
    matcher = SequenceMatcher(None, old_words, new_words)
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.append(" ".join(old_words[i1:i2]))
        elif tag == "delete":
            result.append(
                f"<span style='background-color:salmon;text-decoration:line-through;'>{' '.join(old_words[i1:i2])}</span>"
            )
        elif tag == "insert":
            result.append(
                f"<span style='background-color:lightgreen;'>{' '.join(new_words[j1:j2])}</span>"
            )
        elif tag == "replace":
            result.append(
                f"<span style='background-color:salmon;text-decoration:line-through;'>{' '.join(old_words[i1:i2])}</span>"
            )
            result.append(
                f"<span style='background-color:lightgreen;'>{' '.join(new_words[j1:j2])}</span>"
            )
    return " ".join(result)

# Create tabs
tab_json, tab_csv, tab_utils, tab_analysis, tab_report = st.tabs([
    "JSON Analyzer", "CSV Reviewer", "Utils", "Analysis", "Error Analyzer"
])

### JSON Analyzer ###
with tab_json:
    st.title("JSON Summary Analyzer")
    uploaded_file = st.file_uploader("Upload JSON file", type=["json"], key="uploader")
    if uploaded_file:
        data = json.load(uploaded_file)
        st.sidebar.header("Navigation")
        idx = st.sidebar.number_input("Index", 0, len(data)-1, 0, 1, key="entry_idx")
        entry = data[idx]

        st.header("Source Article")
        st.text_area("Source Article", entry.get("source_article", ""), height=200, key=f"src{idx}", label_visibility="hidden")

        st.header("Gold Summary")
        gold = entry.get("gold_summary", "")
        st.text_area("Gold Summary", gold, height=150, key=f"gold{idx}", label_visibility="hidden")

        corrupted = entry.get("correctness_corrupted_summaries", [])
        if corrupted:
            sel = st.selectbox("Corrupted #", range(len(corrupted)), key="cidx")
            corr = corrupted[sel]

            st.subheader("Error Types")
            if "error_types" in corr:
                errs = corr.get("error_types", [])
            else:
                errs = {t["type"] for change in corr.get("transformation_list", []) for t in change.get("applied_transformations", [])}
            st.write(", ".join(sorted(errs)) if errs else "No error types found.")

            st.subheader("Diff")
            st.markdown(highlight_diff(gold, corr.get("corrupted_summary", "")), unsafe_allow_html=True)

        if st.sidebar.checkbox("Show Raw JSON", key="rawjson"):
            st.json(entry)

### CSV Reviewer ###
with tab_csv:
    st.title("CSV Reviewer")
    csv_file = st.file_uploader(
        "Upload CSV (premise, hypothesis, label[, error_types])", type=["csv"], key="csv_uploader"
    )
    if csv_file:
        df = pd.read_csv(csv_file)
        has_err = 'error_types' in df.columns
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True)

        # Custom comparison
        st.subheader("Custom Text Comparison")
        a = st.text_area("Text A", key="ta")
        b = st.text_area("Text B", key="tb")
        if a and b:
            st.markdown(highlight_diff(a, b), unsafe_allow_html=True)

        # Inspect a row (RTL)
        st.subheader("Inspect a Row")
        ridx = st.number_input("בחר שורה", 0, len(df)-1, 0, 1, key="ridx")
        row = df.iloc[ridx]
        st.markdown(f"<div dir='rtl'>Premise: {row['premise']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div dir='rtl'>Hypothesis: {row['hypothesis']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div dir='rtl'>Label: {row['label']}</div>", unsafe_allow_html=True)
        if has_err:
            st.markdown(f"<div dir='rtl'>Error Types: {row['error_types']}</div>", unsafe_allow_html=True)

        # Label & error stats
        st.subheader("Label Distribution")
        st.bar_chart(df['label'].value_counts())
        if has_err:
            st.subheader("Error Types Distribution")
            exploded = df['error_types'].dropna().str.split(',', expand=True).stack().str.strip()
            st.bar_chart(exploded.value_counts())

        # Word length histogram
        st.subheader("Word Length Histogram")
        premise_lens = df['premise'].str.split().str.len().value_counts().sort_index()
        hyp_lens = df['hypothesis'].str.split().str.len().value_counts().sort_index()
        st.bar_chart(premise_lens)
        st.bar_chart(hyp_lens)

### Utils ###
with tab_utils:
    st.title("Utils")
    # Combine CSV
    st.subheader("Combine CSV Files")
    csvs = st.file_uploader("CSV to combine", type=["csv"], accept_multiple_files=True, key="ccomb")
    if csvs:
        dfc = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
        st.download_button("Download CSV", dfc.to_csv(index=False).encode('utf-8'), "combined.csv")
    # Combine JSON
    st.subheader("Combine JSON Files")
    jsns = st.file_uploader("JSON to combine", type=["json"], accept_multiple_files=True, key="jcomb")
    if jsns:
        lst = []
        for f in jsns:
            lst.extend(json.load(f))
        st.download_button("Download JSON", json.dumps(lst, ensure_ascii=False), "combined.json")
    # Split Data
    st.subheader("Split Data")
    if 'df' in locals():
        split_mode = st.selectbox("Mode", ["kfold", "naive"], key="mode")
        df2 = df.copy()
        if split_mode == 'kfold':
            k = st.number_input("K folds", 2, 20, 5, key="k")
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            for i, (train_idx, test_idx) in enumerate(skf.split(df2, df2['label'])):
                tr, te = df2.iloc[train_idx], df2.iloc[test_idx]
                st.download_button(f"train_{i}.csv", tr.to_csv(index=False).encode(), f"train_{i}.csv")
                st.download_button(f"test_{i}.csv", te.to_csv(index=False).encode(), f"test_{i}.csv")
        else:
            pval = st.slider("Validation %", 0.0, 0.5, 0.1, key="pval")
            ptr = st.slider("Test %", 0.0, 0.5, 0.1, key="ptr")
            train, temp = train_test_split(df2, test_size=(pval+ptr), stratify=df2['label'], random_state=42)
            val, test = train_test_split(temp, test_size=ptr/(pval+ptr), stratify=temp['label'], random_state=42)
            st.download_button("train.csv", train.to_csv(index=False).encode(), "train.csv")
            st.download_button("val.csv", val.to_csv(index=False).encode(), "val.csv")
            st.download_button("test.csv", test.to_csv(index=False).encode(), "test.csv")

### Analysis ###
with tab_analysis:
    st.title("CSV Analysis")
    csv_file_analysis = st.file_uploader("Upload CSV for Analysis", type=["csv"], key="analysis_uploader")
    if csv_file_analysis:
        df_an = pd.read_csv(csv_file_analysis)

        # Unique counts & exact match
        unique_p = df_an['premise'].nunique()
        unique_h = df_an['hypothesis'].nunique()
        unique_pairs = df_an[['premise','hypothesis']].drop_duplicates().shape[0]
        clean = lambda s: re.sub(r"\W+","",str(s)).lower()
        exact = (df_an['premise'].fillna('').apply(clean) == df_an['hypothesis'].fillna('').apply(clean)).sum()
        st.write(f"Unique premises: {unique_p}, hypotheses: {unique_h}, pairs: {unique_pairs}, exact matches: {exact}")

        # Length stats and distributions
        p_len = df_an['premise'].str.split().str.len()
        h_len = df_an['hypothesis'].str.split().str.len()
        st.write("Premise length → mean:{:.1f}, min:{}, max:{}".format(p_len.mean(), p_len.min(), p_len.max()))
        st.write("Hypothesis length → mean:{:.1f}, min:{}, max:{}".format(h_len.mean(), h_len.min(), h_len.max()))
        st.subheader("Select Sample to View")
        sample_type = st.selectbox("Category",[
            "Unique Premises",
            "Unique Hypotheses",
            "Unique P-H Pairs",
            "Exact Matches",
            "Non-Matches"
        ])
        if sample_type == "Unique Premises":
            samples = df_an['premise'].drop_duplicates().sample(min(5, unique_p))
        elif sample_type == "Unique Hypotheses":
            samples = df_an['hypothesis'].drop_duplicates().sample(min(5, unique_h))
        elif sample_type == "Unique P-H Pairs":
            samples = df_an[['premise','hypothesis']].drop_duplicates().sample(min(5, unique_pairs))
        elif sample_type == "Exact Matches":
            mask = df_an['premise'].fillna('').apply(clean) == df_an['hypothesis'].fillna('').apply(clean)
            samples = df_an[mask].head(5)
        else:
            mask = df_an['premise'].fillna('').apply(clean) != df_an['hypothesis'].fillna('').apply(clean)
            samples = df_an[mask].head(5)
        st.dataframe(samples)

with tab_report:
    st.title("Correctness Report Analyzer")
    report_file = st.file_uploader(
        "Upload finder JSON report",
        type=["json"],
        key="report_uploader"
    )
    if report_file:
        data = json.load(report_file)
        df = pd.DataFrame(data)

        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True)

        # error‐name distribution
        st.subheader("Error Name Distribution")
        st.bar_chart(df["correctness_error_name"].value_counts())

        # distribution by source/doc_id if you like
        st.subheader("Entries per Source")
        st.bar_chart(df["source"].value_counts())

        # Inspect a single record
        st.subheader("Inspect a Record")
        idx = st.number_input(
            "Record index", 0, len(df) - 1, 0, 1, key="report_idx"
        )
        rec = df.iloc[idx]
        st.markdown(f"**Doc_ID:**  {rec['doc_id']}")
        st.markdown(f"**Premise:**  {rec['premise']}")
        st.markdown(f"**AI Summary:**  {rec['ai_summary']}")
        #st.markdown(f"**generative_summary:**  {rec['generative_summary']}")
        st.markdown(f"**Error Name:**  {rec['correctness_error_name']}")
        st.markdown(f"**Reasoning:**  {rec['correctness_error_reasoning']}")
        st.markdown(f"**Features:**  {rec['mrl_unique_hebrew_features']}")
        st.markdown(f"**Why_its_factual_error:**  {rec['why_its_factual_error']}")
        st.markdown(f"**Evidence (Summary):**  {rec['evidence_from_summary']}")
        st.markdown(f"**Evidence (Source):**  {rec['evidence_from_source_article']}")
        st.markdown(f"**severity**  {rec['severity']}")