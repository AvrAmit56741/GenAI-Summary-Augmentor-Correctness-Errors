import streamlit as st
import json
import pandas as pd
from difflib import SequenceMatcher
import re
from sklearn.model_selection import train_test_split, StratifiedKFold

st.set_page_config(page_title="Summary & CSV Analyzer & Utils", layout="wide")

# Initialize session state for error type filtering
if 'selected_error_types' not in st.session_state:
    st.session_state.selected_error_types = []
if 'quality_ratings' not in st.session_state:
    st.session_state.quality_ratings = {}

# Helper to highlight word‐level diffs with dark theme support
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
                f"<span style='background-color:#8b0000;color:white;text-decoration:line-through;padding:2px;'>{' '.join(old_words[i1:i2])}</span>"
            )
        elif tag == "insert":
            result.append(
                f"<span style='background-color:#006400;color:white;padding:2px;'>{' '.join(new_words[j1:j2])}</span>"
            )
        elif tag == "replace":
            result.append(
                f"<span style='background-color:#8b0000;color:white;text-decoration:line-through;padding:2px;'>{' '.join(old_words[i1:i2])}</span>"
            )
            result.append(
                f"<span style='background-color:#006400;color:white;padding:2px;'>{' '.join(new_words[j1:j2])}</span>"
            )
    return " ".join(result)

# Helper to parse error types from string
def parse_error_types(error_str):
    if pd.isna(error_str) or error_str == '':
        return []
    return [error.strip() for error in str(error_str).split(',') if error.strip()]

# Helper to get all unique error types from dataframe
def get_all_error_types(df):
    all_errors = []
    if 'error_types' in df.columns:
        for error_str in df['error_types'].dropna():
            all_errors.extend(parse_error_types(error_str))
    return sorted(list(set(all_errors)))

# Helper to filter dataframe by selected error types
def filter_by_error_types(df, selected_types):
    if not selected_types or 'error_types' not in df.columns:
        return df
    
    mask = df['error_types'].apply(
        lambda x: any(error in parse_error_types(x) for error in selected_types) if pd.notna(x) else False
    )
    return df[mask]

# Helper to extract claim transformations from transformation_list
def extract_claim_transformations(transformation_list_str):
    """Extract claim-level transformations from the transformation_list JSON string"""
    try:
        if pd.isna(transformation_list_str):
            return []
        
        transformation_list = json.loads(transformation_list_str) if isinstance(transformation_list_str, str) else transformation_list_str
        
        claims_with_transformations = []
        for transformation in transformation_list:
            if isinstance(transformation, dict):
                original_claim = transformation.get('original_claim', '')
                transformed_claim = transformation.get('transformed_claim', '')
                applied_transformations = transformation.get('applied_transformations', [])
                
                for applied_trans in applied_transformations:
                    claims_with_transformations.append({
                        'original_claim': original_claim,
                        'transformed_claim': transformed_claim,
                        'error_type': applied_trans.get('type', 'unknown'),
                        'transformation_details': applied_trans
                    })
        
        return claims_with_transformations
    except:
        return []

# Helper to get claims by error type from entire dataset (CSV format)
def get_claims_by_error_type(df, error_type):
    """Extract all claims with specific error type from the dataset"""
    all_claims = []
    
    for idx, row in df.iterrows():
        # Check if transformation_list exists
        if 'transformation_list' in df.columns and pd.notna(row['transformation_list']):
            claims = extract_claim_transformations(row['transformation_list'])
            for claim in claims:
                if claim['error_type'] == error_type:
                    claim['doc_id'] = row.get('doc_id', f'row_{idx}')
                    claim['source_row'] = idx
                    all_claims.append(claim)
    
    return all_claims

# Helper to get claims by error type from JSON dataset
def get_claims_by_error_type_json(json_data, error_type):
    """Extract all claims with specific error type from JSON dataset"""
    all_claims = []
    
    for doc_idx, doc in enumerate(json_data):
        doc_id = f"doc_{doc_idx}"
        
        # Look through each corrupted summary
        for summary_idx, corrupted_summary in enumerate(doc.get('correctness_corrupted_summaries', [])):
            transformation_list = corrupted_summary.get('transformation_list', [])
            
            for transformation in transformation_list:
                if isinstance(transformation, dict):
                    original_claim = transformation.get('original_claim', '')
                    transformed_claim = transformation.get('transformed_claim', '')
                    applied_transformations = transformation.get('applied_transformations', [])
                    
                    for applied_trans in applied_transformations:
                        if applied_trans.get('type', '') == error_type:
                            all_claims.append({
                                'original_claim': original_claim,
                                'transformed_claim': transformed_claim,
                                'error_type': error_type,
                                'transformation_details': applied_trans,
                                'doc_id': doc_id,
                                'summary_idx': summary_idx,
                                'source_article': doc.get('source_article', ''),
                                'gold_summary': doc.get('gold_summary', '')
                            })
    
    return all_claims

# Helper to get all error types from JSON data
def get_all_error_types_json(json_data):
    """Extract all unique error types from JSON dataset"""
    all_errors = set()
    
    for doc in json_data:
        for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
            # Check transformation_list
            for transformation in corrupted_summary.get('transformation_list', []):
                for applied_trans in transformation.get('applied_transformations', []):
                    error_type = applied_trans.get('type', '')
                    if error_type:
                        all_errors.add(error_type)
            
            # Also check distinct_errors_type_names if available
            for error_type in corrupted_summary.get('distinct_errors_type_names', []):
                all_errors.add(error_type)
    
    return sorted(list(all_errors))

# Helper to filter JSON data by error types
def filter_json_by_error_types(json_data, selected_types):
    """Filter JSON data to only include entries with selected error types"""
    if not selected_types:
        return json_data
    
    filtered_data = []
    for doc in json_data:
        filtered_summaries = []
        
        for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
            # Check if this summary has any of the selected error types
            has_selected_error = False
            
            # Check transformation_list
            for transformation in corrupted_summary.get('transformation_list', []):
                for applied_trans in transformation.get('applied_transformations', []):
                    if applied_trans.get('type', '') in selected_types:
                        has_selected_error = True
                        break
                if has_selected_error:
                    break
            
            # Also check distinct_errors_type_names
            if not has_selected_error:
                for error_type in corrupted_summary.get('distinct_errors_type_names', []):
                    if error_type in selected_types:
                        has_selected_error = True
                        break
            
            if has_selected_error:
                filtered_summaries.append(corrupted_summary)
        
        if filtered_summaries:
            filtered_doc = doc.copy()
            filtered_doc['correctness_corrupted_summaries'] = filtered_summaries
            filtered_data.append(filtered_doc)
    
    return filtered_data

# Create tabs
tab_json, tab_csv, tab_utils, tab_analysis, tab_report, tab_logs, tab_dataset = st.tabs([
    "JSON Analyzer", "CSV Reviewer", "Utils", "Analysis", "Error Analyzer", "API Logs", "Dataset Analysis"
])

### JSON Analyzer ###
with tab_json:
    st.title("Enhanced JSON Summary Analyzer")
    st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload JSON file", type=["json"], key="uploader")
    if uploaded_file:
        data = json.load(uploaded_file)
        
        # Initialize JSON-specific session state
        if 'json_selected_error_types' not in st.session_state:
            st.session_state.json_selected_error_types = []
        if 'json_quality_ratings' not in st.session_state:
            st.session_state.json_quality_ratings = {}
        
        # Get all error types from JSON data
        all_error_types = get_all_error_types_json(data)
        
        # Sidebar for filtering and navigation
        st.sidebar.header("Filters & Navigation")
        
        # Error type filtering
        if all_error_types:
            st.sidebar.subheader("Filter by Error Types")
            
            # Ensure default values exist in current options
            valid_defaults = [et for et in st.session_state.json_selected_error_types if et in all_error_types]
            
            selected_error_types = st.sidebar.multiselect(
                "Select error types to filter:",
                options=all_error_types,
                default=valid_defaults,
                key="json_error_type_filter"
            )
            st.session_state.json_selected_error_types = selected_error_types
            
            # Apply filters
            filtered_data = filter_json_by_error_types(data, selected_error_types)
            
            # Clear filters button
            if st.sidebar.button("Clear All Filters", key="json_clear_filters"):
                st.session_state.json_selected_error_types = []
                st.experimental_rerun()
        else:
            filtered_data = data
            selected_error_types = []
            st.sidebar.markdown("⚠️ No error types found in data")
        
        # Quick stats panel
        total_docs = len(data)
        filtered_docs = len(filtered_data)
        total_corrupted = sum(len(doc.get('correctness_corrupted_summaries', [])) for doc in data)
        filtered_corrupted = sum(len(doc.get('correctness_corrupted_summaries', [])) for doc in filtered_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", total_docs)
        with col2:
            st.metric("Filtered Documents", filtered_docs)
        with col3:
            st.metric("Corrupted Summaries", f"{filtered_corrupted}/{total_corrupted}")
        
        # Interactive Error Types Analysis
        if all_error_types:
            st.subheader("🎯 Interactive Error Types Distribution")
            
            # Create error type counts
            error_counts = {}
            for doc in data:
                for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
                    for error_type in corrupted_summary.get('distinct_errors_type_names', []):
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            if error_counts:
                error_df = pd.DataFrame(list(error_counts.items()), columns=['Error Type', 'Count'])
                error_df = error_df.sort_values('Count', ascending=False)
                
                # Display error types as clickable buttons
                st.markdown("<div style='color: white;'><strong>Click on error types below to filter dataset:</strong></div>", unsafe_allow_html=True)
                
                # Create columns for error type buttons
                cols = st.columns(3)
                for i, (error_type, count) in enumerate(error_df.values):
                    col = cols[i % 3]
                    with col:
                        button_style = "🔹" if error_type not in st.session_state.json_selected_error_types else "✅"
                        if st.button(f"{button_style} {error_type} ({count})", key=f"json_error_btn_{error_type}"):
                            if error_type in st.session_state.json_selected_error_types:
                                st.session_state.json_selected_error_types.remove(error_type)
                            else:
                                st.session_state.json_selected_error_types.append(error_type)
                            st.experimental_rerun()
                
                # Show bar chart for visualization
                st.bar_chart(error_df.set_index('Error Type')['Count'])
                
                # Claim-Level Analysis for Selected Error Types
                if selected_error_types:
                    st.subheader(f"🔍 Claim-Level Analysis: {', '.join(selected_error_types)}")
                    
                    # Get all claims for selected error types
                    all_claims_for_error_types = []
                    for error_type in selected_error_types:
                        claims = get_claims_by_error_type_json(data, error_type)
                        all_claims_for_error_types.extend(claims)
                    
                    if all_claims_for_error_types:
                        st.markdown(f"<div style='color: white;'><strong>Found {len(all_claims_for_error_types)} claim transformations</strong></div>", unsafe_allow_html=True)
                        
                        # Display claims in expandable table
                        st.subheader("📋 All Claims with Selected Error Types")
                        
                        for i, claim in enumerate(all_claims_for_error_types):
                            with st.expander(f"Claim {i+1}: {claim['error_type']} ({claim['doc_id']})"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("<div style='color: white;'><strong>🟢 Original Claim:</strong></div>", unsafe_allow_html=True)
                                    st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 10px; border-radius: 5px; border: 1px solid #4CAF50;'>{claim['original_claim']}</div>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("<div style='color: white;'><strong>🔴 Transformed Claim:</strong></div>", unsafe_allow_html=True)
                                    st.markdown(f"<div dir='rtl' style='background-color: #4d1a1a; color: white; padding: 10px; border-radius: 5px; border: 1px solid #f44336;'>{claim['transformed_claim']}</div>", unsafe_allow_html=True)
                                
                                # Show diff between original and transformed claim
                                st.markdown("<div style='color: white;'><strong>⚡ Transformation Diff:</strong></div>", unsafe_allow_html=True)
                                diff_html = highlight_diff(claim['original_claim'], claim['transformed_claim'])
                                st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 10px; border-radius: 5px; border: 1px solid #666;'>{diff_html}</div>", unsafe_allow_html=True)
                                
                                # Show transformation details
                                st.markdown("<div style='color: white;'><strong>🔧 Transformation Details:</strong></div>", unsafe_allow_html=True)
                                details = claim.get('transformation_details', {})
                                if 'diffs' in details:
                                    for diff in details['diffs']:
                                        st.markdown(f"<div style='color: #ffd700;'>• Changed '{diff.get('orig_text', '')}' → '{diff.get('trans_text', '')}' (positions {diff.get('orig_start', '')}-{diff.get('orig_end', '')})</div>", unsafe_allow_html=True)
                                elif 'original' in details and 'replacement' in details:
                                    st.markdown(f"<div style='color: #ffd700;'>• Replaced '{details['original']}' → '{details['replacement']}'</div>", unsafe_allow_html=True)
                                
                                # Show context
                                st.markdown("<div style='color: white;'><strong>📄 Context:</strong></div>", unsafe_allow_html=True)
                                st.markdown(f"<div style='color: #ccc; font-size: 0.9em;'>Document: {claim['doc_id']} | Summary: {claim['summary_idx']}</div>", unsafe_allow_html=True)
                                
                                # Quality rating for this specific claim transformation
                                rating_key = f"json_claim_quality_{i}_{claim['error_type']}_{hash(claim['transformed_claim'])}"
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    if st.button("👍 Good", key=f"json_good_{rating_key}"):
                                        st.session_state.json_quality_ratings[rating_key] = "good"
                                        st.success("✅")
                                
                                with col2:
                                    if st.button("👎 Poor", key=f"json_poor_{rating_key}"):
                                        st.session_state.json_quality_ratings[rating_key] = "poor"
                                        st.error("❌")
                                
                                with col3:
                                    if st.button("❓ Unclear", key=f"json_unclear_{rating_key}"):
                                        st.session_state.json_quality_ratings[rating_key] = "unclear"
                                        st.warning("❓")
                                
                                with col4:
                                    current_rating = st.session_state.json_quality_ratings.get(rating_key, "Not rated")
                                    st.markdown(f"<div style='color: #ffd700;'>Status: {current_rating}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color: white;'>⚠️ No claims found for selected error types</div>", unsafe_allow_html=True)
        
        # Document Navigation and Summary Comparison
        st.subheader("📄 Document Navigation & Summary Comparison")
        
        # Document selector
        doc_idx = st.sidebar.number_input("Document Index", 0, len(filtered_data)-1, 0, 1, key="json_doc_idx")
        
        if len(filtered_data) > 0:
            entry = filtered_data[doc_idx]
            
            # Show source article in expandable section
            with st.expander("📰 Source Article", expanded=False):
                st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 15px; border-radius: 5px;'>{entry.get('source_article', '')}</div>", unsafe_allow_html=True)
            
            # Show gold summary
            st.markdown("<div style='color: white;'><strong>🏆 Gold Summary:</strong></div>", unsafe_allow_html=True)
            gold = entry.get("gold_summary", "")
            st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #4CAF50;'>{gold}</div>", unsafe_allow_html=True)
            
            # Show corrupted summaries
            corrupted = entry.get("correctness_corrupted_summaries", [])
            if corrupted:
                st.markdown("<div style='color: white;'><strong>🔍 Corrupted Summaries vs Gold:</strong></div>", unsafe_allow_html=True)
                
                for i, corr in enumerate(corrupted):
                    # Get error types for this summary
                    error_types = corr.get('distinct_errors_type_names', [])
                    if not error_types:
                        error_types = {t["type"] for change in corr.get("transformation_list", []) for t in change.get("applied_transformations", [])}
                    
                    error_types_str = ", ".join(sorted(error_types)) if error_types else "No errors"
                    
                    with st.expander(f"Corrupted Summary {i+1}: {error_types_str}"):
                        # Show corrupted summary vs gold diff
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("<div style='color: white;'><strong>🔴 Corrupted Summary:</strong></div>", unsafe_allow_html=True)
                            corrupted_text = corr.get("corrupted_summary", "")
                            st.markdown(f"<div dir='rtl' style='background-color: #4d1a1a; color: white; padding: 10px; border-radius: 5px;'>{corrupted_text}</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div style='color: white;'><strong>Diff vs Gold:</strong></div>", unsafe_allow_html=True)
                            diff_html = highlight_diff(gold, corrupted_text)
                            st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 10px; border-radius: 5px;'>{diff_html}</div>", unsafe_allow_html=True)
                        
                        # Show transformation metadata
                        st.markdown("<div style='color: white;'><strong>📊 Summary Metadata:</strong></div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='color: #ffd700;'>• Transformations Performed: {corr.get('transformations_performed', 0)}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='color: #ffd700;'>• Transformation Score: {corr.get('transformation_score', 0):.3f}</div>", unsafe_allow_html=True)
                        
                        # Show claim-level transformations
                        transformation_list = corr.get('transformation_list', [])
                        if transformation_list:
                            st.markdown("<div style='color: white;'><strong>🔧 Claim-Level Transformations:</strong></div>", unsafe_allow_html=True)
                            
                            for j, transformation in enumerate(transformation_list):
                                st.markdown(f"<div style='color: #ffd700; font-weight: bold; margin: 10px 0;'>📋 Claim Transformation {j+1}:</div>", unsafe_allow_html=True)
                                
                                original_claim = transformation.get('original_claim', '')
                                transformed_claim = transformation.get('transformed_claim', '')
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("<div style='color: white;'><strong>Original:</strong></div>", unsafe_allow_html=True)
                                    st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 8px; border-radius: 5px; margin: 5px 0;'>{original_claim}</div>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("<div style='color: white;'><strong>Transformed:</strong></div>", unsafe_allow_html=True)
                                    st.markdown(f"<div dir='rtl' style='background-color: #4d1a1a; color: white; padding: 8px; border-radius: 5px; margin: 5px 0;'>{transformed_claim}</div>", unsafe_allow_html=True)
                                
                                # Show claim diff
                                st.markdown("<div style='color: white;'><strong>Diff:</strong></div>", unsafe_allow_html=True)
                                claim_diff = highlight_diff(original_claim, transformed_claim)
                                st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 8px; border-radius: 5px; margin: 5px 0; border: 1px solid #666;'>{claim_diff}</div>", unsafe_allow_html=True)
                                
                                # Show applied transformations
                                applied_transformations = transformation.get('applied_transformations', [])
                                if applied_transformations:
                                    st.markdown("<div style='color: white;'><strong>Applied Transformations:</strong></div>", unsafe_allow_html=True)
                                    for k, applied_trans in enumerate(applied_transformations):
                                        st.markdown(f"<div style='color: #ffd700;'>• {applied_trans.get('type', 'unknown')}</div>", unsafe_allow_html=True)
                                        if 'original' in applied_trans and 'replacement' in applied_trans:
                                            st.markdown(f"<div style='color: #ccc; margin-left: 20px;'>'{applied_trans['original']}' → '{applied_trans['replacement']}'</div>", unsafe_allow_html=True)
                                        elif 'diffs' in applied_trans:
                                            for diff in applied_trans['diffs']:
                                                st.markdown(f"<div style='color: #ccc; margin-left: 20px;'>Position {diff.get('orig_start', '')}-{diff.get('orig_end', '')}: '{diff.get('orig_text', '')}' → '{diff.get('trans_text', '')}'</div>", unsafe_allow_html=True)
                                
                                # Add separator between claims
                                st.markdown("<hr style='border-color: #444; margin: 15px 0;'>", unsafe_allow_html=True)
        
        # Document-level summary comparison (same doc, different summaries)
        if len(filtered_data) > 0:
            st.subheader("📄 Document Summary Comparison")
            
            current_entry = filtered_data[doc_idx] if doc_idx < len(filtered_data) else filtered_data[0]
            corrupted_summaries = current_entry.get("correctness_corrupted_summaries", [])
            
            if corrupted_summaries:
                st.markdown("<div style='color: white;'><strong>Compare all summaries for this document:</strong></div>", unsafe_allow_html=True)
                
                # Show gold summary
                st.markdown("<div style='color: white;'><strong>🏆 Gold Summary:</strong></div>", unsafe_allow_html=True)
                gold = current_entry.get("gold_summary", "")
                st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #4CAF50;'>{gold}</div>", unsafe_allow_html=True)
                
                # Show each corrupted summary
                for i, corrupted_summary in enumerate(corrupted_summaries):
                    error_types = corrupted_summary.get('distinct_errors_type_names', [])
                    error_types_str = ", ".join(error_types) if error_types else "Unknown errors"
                    
                    with st.expander(f"Corrupted Summary {i+1}: {error_types_str}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("<div style='color: white;'><strong>Corrupted Text:</strong></div>", unsafe_allow_html=True)
                            corrupted_text = corrupted_summary.get("corrupted_summary", "")
                            st.markdown(f"<div dir='rtl' style='background-color: #4d1a1a; color: white; padding: 10px; border-radius: 5px;'>{corrupted_text}</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div style='color: white;'><strong>Diff vs Gold:</strong></div>", unsafe_allow_html=True)
                            diff_html = highlight_diff(gold, corrupted_text)
                            st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 10px; border-radius: 5px;'>{diff_html}</div>", unsafe_allow_html=True)
        
        # Export functionality
        if selected_error_types:
            st.subheader("📥 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export filtered JSON data
                filtered_json = json.dumps(filtered_data, ensure_ascii=False, indent=2).encode('utf-8')
                st.download_button(
                    "📥 Download Filtered JSON",
                    filtered_json,
                    f"filtered_json_{'-'.join(selected_error_types[:2])}.json",
                    "application/json"
                )
            
            with col2:
                # Export quality ratings
                if st.session_state.json_quality_ratings:
                    ratings_df = pd.DataFrame([
                        {"example_id": k, "rating": v} 
                        for k, v in st.session_state.json_quality_ratings.items()
                    ])
                    ratings_csv = ratings_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 Download Quality Ratings",
                        ratings_csv,
                        "json_quality_ratings.csv",
                        "text/csv"
                    )
        
        # Statistics Dashboard
        st.subheader("📊 Dataset Statistics")
        
        if len(filtered_data) > 0:
            # Error type distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div style='color: white;'><strong>Error Types Distribution</strong></div>", unsafe_allow_html=True)
                if error_counts:
                    filtered_error_counts = {}
                    for doc in filtered_data:
                        for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
                            for error_type in corrupted_summary.get('distinct_errors_type_names', []):
                                filtered_error_counts[error_type] = filtered_error_counts.get(error_type, 0) + 1
                    
                    if filtered_error_counts:
                        filtered_error_df = pd.DataFrame(list(filtered_error_counts.items()), columns=['Error Type', 'Count'])
                        st.bar_chart(filtered_error_df.set_index('Error Type')['Count'])
            
            with col2:
                st.markdown("<div style='color: white;'><strong>Transformation Scores</strong></div>", unsafe_allow_html=True)
                scores = []
                for doc in filtered_data:
                    for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
                        score = corrupted_summary.get('transformation_score', 0)
                        if score > 0:
                            scores.append(score)
                
                if scores:
                    scores_df = pd.DataFrame({'Transformation Score': scores})
                    st.bar_chart(scores_df['Transformation Score'].value_counts().sort_index())
        
        # Raw JSON view (moved to sidebar)
        if st.sidebar.checkbox("Show Raw JSON", key="rawjson"):
            current_entry = filtered_data[doc_idx] if doc_idx < len(filtered_data) else filtered_data[0]
            st.json(current_entry)

### API Logs Analyzer ###
with tab_logs:
    st.title("🔍 API Logs Analyzer")
    st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)
    
    uploaded_log_file = st.file_uploader("Upload API Log File (.jsonl)", type=["jsonl"], key="log_uploader")
    
    if uploaded_log_file:
        try:
            # Read JSONL file
            log_entries = []
            for line in uploaded_log_file:
                line_str = line.decode('utf-8').strip()
                if line_str:
                    try:
                        # Handle standard logger format: "timestamp - logger_name - thread - level - message"
                        if ' - INFO - ' in line_str or ' - ERROR - ' in line_str or ' - DEBUG - ' in line_str:
                            # Split by the last ' - ' to get the message part
                            parts = line_str.split(' - ')
                            if len(parts) >= 4:
                                timestamp = parts[0]
                                logger_name = parts[1] 
                                thread_name = parts[2]
                                level = parts[3]
                                message = ' - '.join(parts[4:])  # Rejoin in case message contains ' - '
                                
                                # Extract log type and JSON from message
                                if message.startswith('REQUEST: ') or message.startswith('RESPONSE: ') or message.startswith('ERROR: '):
                                    log_type = message.split(': ')[0]
                                    json_part = message[len(log_type) + 2:]
                                    
                                    try:
                                        parsed_json = json.loads(json_part)
                                        
                                        # Handle case where parsed_json might be a string or other type
                                        if isinstance(parsed_json, dict):
                                            entry = parsed_json.copy()
                                            entry.update({
                                                'log_type': log_type,
                                                'timestamp': timestamp,
                                                'logger_name': logger_name,
                                                'thread_name': thread_name,
                                                'level': level
                                            })
                                        else:
                                            # If it's not a dict, wrap it
                                            entry = {
                                                'log_type': log_type,
                                                'timestamp': timestamp,
                                                'logger_name': logger_name,
                                                'thread_name': thread_name,
                                                'level': level,
                                                'data': parsed_json
                                            }
                                        log_entries.append(entry)
                                    except json.JSONDecodeError:
                                        # If JSON parsing fails, store as raw message
                                        log_entries.append({
                                            'log_type': 'RAW',
                                            'timestamp': timestamp,
                                            'logger_name': logger_name,
                                            'thread_name': thread_name,
                                            'level': level,
                                            'raw_message': message
                                        })
                                else:
                                    # Regular log message without REQUEST/RESPONSE/ERROR prefix
                                    log_entries.append({
                                        'log_type': 'LOG',
                                        'timestamp': timestamp,
                                        'logger_name': logger_name,
                                        'thread_name': thread_name,
                                        'level': level,
                                        'message': message
                                    })
                        
                        # Handle direct JSON lines (fallback)
                        elif line_str.startswith('{') and line_str.endswith('}'):
                            entry = json.loads(line_str)
                            log_entries.append(entry)
                            
                    except json.JSONDecodeError:
                        continue
            
            if log_entries:
                st.success(f"✅ Loaded {len(log_entries)} log entries")
                
                # Quick Statistics
                st.subheader("📊 Log Statistics")
                
                # Count by type
                log_types = {}
                models_used = {}
                errors_count = 0
                
                for entry in log_entries:
                    # Count log types
                    log_type = entry.get('log_type', 'UNKNOWN')
                    log_types[log_type] = log_types.get(log_type, 0) + 1
                    
                    # Count models
                    if 'model' in entry:
                        model = entry['model']
                        models_used[model] = models_used.get(model, 0) + 1
                    
                    # Count errors
                    if log_type == 'ERROR' or 'error' in entry:
                        errors_count += 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Entries", len(log_entries))
                with col2:
                    st.metric("Errors", errors_count)
                with col3:
                    st.metric("Models Used", len(models_used))
                
                # Log Type Distribution
                if log_types:
                    st.subheader("📈 Log Type Distribution")
                    st.bar_chart(pd.Series(log_types))
                
                # Model Usage
                if models_used:
                    st.subheader("🤖 Model Usage")
                    st.bar_chart(pd.Series(models_used))
                
                # Log Entry Browser
                st.subheader("🔍 Browse Log Entries")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    filter_type = st.selectbox("Filter by Type:", ["All"] + list(log_types.keys()))
                with col2:
                    show_errors_only = st.checkbox("Show Errors Only")
                
                # Apply filters
                filtered_entries = log_entries
                if filter_type != "All":
                    filtered_entries = [e for e in filtered_entries if e.get('log_type') == filter_type]
                if show_errors_only:
                    filtered_entries = [e for e in filtered_entries if e.get('log_type') == 'ERROR' or 'error' in e]
                
                st.write(f"Showing {len(filtered_entries)} entries")
                
                # Entry navigator
                if filtered_entries:
                    # Initialize navigation state if not exists
                    if 'nav_entry_idx' not in st.session_state:
                        st.session_state.nav_entry_idx = 0
                    
                    # Ensure nav_entry_idx is within bounds
                    if st.session_state.nav_entry_idx >= len(filtered_entries):
                        st.session_state.nav_entry_idx = 0
                    if st.session_state.nav_entry_idx < 0:
                        st.session_state.nav_entry_idx = 0
                    
                    entry_idx = st.number_input("Entry #", 0, len(filtered_entries)-1, st.session_state.nav_entry_idx, 1, key="log_entry_idx")
                    
                    # Update nav state when number input changes
                    if entry_idx != st.session_state.nav_entry_idx:
                        st.session_state.nav_entry_idx = entry_idx
                    
                    current_entry = filtered_entries[entry_idx]
                    
                    # Display entry details
                    log_type = current_entry.get('log_type', 'UNKNOWN')
                    timestamp = current_entry.get('timestamp', 'Unknown')
                    
                    st.markdown(f"**Entry {entry_idx + 1}/{len(filtered_entries)} - Type: {log_type} - Time: {timestamp}**")
                    
                    # Show different views based on log type
                    if log_type == 'REQUEST':
                        with st.expander("📤 Request Details", expanded=True):
                            # Show logger metadata
                            st.markdown(f"**Logger:** {current_entry.get('logger_name', 'Unknown')} | **Thread:** {current_entry.get('thread_name', 'Unknown')}")
                            
                            if isinstance(current_entry, str):
                                st.text_area("Request Data", current_entry, height=200)
                            else:
                                # Handle your current format where request_data is the JSON string
                                if isinstance(current_entry, str):
                                    st.text_area("Request Data", current_entry, height=200)
                                else:
                                    # Show function and model info
                                    function = current_entry.get('function', 'Unknown')
                                    model = current_entry.get('model', 'Unknown')
                                    st.markdown(f"**Function:** {function} | **Model:** {model}")
                                    
                                    # Show request details
                                    if 'request' in current_entry:
                                        req = current_entry['request']
                                        st.markdown("**System Message:**")
                                        st.text_area("System", req.get('system_message', ''), height=150, key=f"sys_{entry_idx}")
                                        st.markdown("**User Message:**")
                                        st.text_area("User", req.get('user_message', ''), height=150, key=f"user_{entry_idx}")
                                    else:
                                        # Handle simple string format from your Gemini logger
                                        if 'data' in current_entry:
                                            st.text_area("Request Data", str(current_entry['data']), height=200, key=f"data_{entry_idx}")
                                        elif 'prompt:' in str(current_entry):
                                            st.text_area("Prompt Data", str(current_entry), height=200, key=f"prompt_{entry_idx}")
                    
                    elif log_type == 'RESPONSE':
                        with st.expander("📥 Response Details", expanded=True):
                            st.markdown(f"**Logger:** {current_entry.get('logger_name', 'Unknown')} | **Thread:** {current_entry.get('thread_name', 'Unknown')}")
                            
                            if 'response' in current_entry:
                                resp = current_entry['response']
                                st.markdown("**Response Content:**")
                                st.text_area("Content", resp.get('content', ''), height=200, key=f"resp_{entry_idx}")
                                
                                if 'usage' in resp:
                                    usage = resp['usage']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Prompt Tokens", usage.get('prompt_tokens', 0))
                                    with col2:
                                        st.metric("Completion Tokens", usage.get('completion_tokens', 0))
                                    with col3:
                                        st.metric("Total Tokens", usage.get('total_tokens', 0))
                                
                                # Show model and status
                                if 'model' in resp:
                                    st.markdown(f"**Model:** {resp.get('model')}")
                                if 'finish_reason' in resp:
                                    st.markdown(f"**Finish Reason:** {resp.get('finish_reason')}")
                            else:
                                # Handle simple response format
                                st.text_area("Response Data", str(current_entry), height=200, key=f"resp_simple_{entry_idx}")
                    
                    elif log_type == 'ERROR':
                        with st.expander("❌ Error Details", expanded=True):
                            st.markdown(f"**Logger:** {current_entry.get('logger_name', 'Unknown')} | **Thread:** {current_entry.get('thread_name', 'Unknown')}")
                            
                            error_msg = current_entry.get('error', 'Unknown error')
                            error_type = current_entry.get('error_type', 'Unknown')
                            function = current_entry.get('function', 'Unknown')
                            
                            st.error(f"**Error Type:** {error_type}")
                            st.markdown(f"**Function:** {function}")
                            st.text_area("Error Message", error_msg, height=100, key=f"err_{entry_idx}")
                    
                    elif log_type == 'LOG':
                        with st.expander("📄 Log Message", expanded=True):
                            st.markdown(f"**Logger:** {current_entry.get('logger_name', 'Unknown')} | **Thread:** {current_entry.get('thread_name', 'Unknown')}")
                            st.markdown(f"**Level:** {current_entry.get('level', 'Unknown')}")
                            st.text_area("Message", current_entry.get('message', ''), height=100, key=f"log_{entry_idx}")
                    
                    elif log_type == 'RAW':
                        with st.expander("📄 Raw Log Entry", expanded=True):
                            st.markdown(f"**Logger:** {current_entry.get('logger_name', 'Unknown')} | **Thread:** {current_entry.get('thread_name', 'Unknown')}")
                            st.text_area("Raw Message", current_entry.get('raw_message', ''), height=150, key=f"raw_{entry_idx}")
                    
                    # Raw JSON view
                    with st.expander("🔧 Raw JSON Data"):
                        st.json(current_entry)
                    
                    # Navigation buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("⏮️ First") and entry_idx > 0:
                            st.session_state.nav_entry_idx = 0
                            st.experimental_rerun()
                    with col2:
                        if st.button("⏪ Previous") and entry_idx > 0:
                            st.session_state.nav_entry_idx = entry_idx - 1
                            st.experimental_rerun()
                    with col3:
                        if st.button("⏩ Next") and entry_idx < len(filtered_entries) - 1:
                            st.session_state.nav_entry_idx = entry_idx + 1
                            st.experimental_rerun()
                
                # Error Analysis
                if errors_count > 0:
                    st.subheader("🚨 Error Analysis")
                    error_entries = [e for e in log_entries if e.get('log_type') == 'ERROR' or 'error' in e]
                    
                    error_types = {}
                    for error_entry in error_entries:
                        error_type = error_entry.get('error_type', 'Unknown')
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    st.bar_chart(pd.Series(error_types))
                    
                    # Show recent errors
                    st.markdown("**Recent Errors:**")
                    for i, error in enumerate(error_entries[-5:]):  # Last 5 errors
                        error_msg = error.get('error', 'Unknown error')[:100]
                        timestamp = error.get('timestamp', 'Unknown')
                        st.markdown(f"- **{timestamp}**: {error_msg}...")
                
                # Export filtered results
                if filtered_entries:
                    st.subheader("📥 Export")
                    
                    # Create summary report
                    summary_data = {
                        'total_entries': len(log_entries),
                        'filtered_entries': len(filtered_entries),
                        'log_types': log_types,
                        'models_used': models_used,
                        'errors_count': errors_count,
                        'filter_applied': filter_type if filter_type != "All" else None,
                        'errors_only': show_errors_only
                    }
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        summary_json = json.dumps(summary_data, indent=2).encode('utf-8')
                        st.download_button(
                            "📊 Download Summary",
                            summary_json,
                            "log_analysis_summary.json",
                            "application/json"
                        )
                    
                    with col2:
                        filtered_json = json.dumps(filtered_entries, ensure_ascii=False, indent=2).encode('utf-8')
                        st.download_button(
                            "📥 Download Filtered Logs",
                            filtered_json,
                            "filtered_api_logs.json",
                            "application/json"
                        )
            
            else:
                st.warning("⚠️ No valid log entries found in the file")
        
        except Exception as e:
            st.error(f"❌ Error reading log file: {str(e)}")
    
    else:
        st.info("📁 Upload a .jsonl API log file to start analysis")
        st.markdown("""
        **Expected log format:**
        - JSONL file with one JSON object per line
        - Supports formats like: `REQUEST: {...}`, `RESPONSE: {...}`, `ERROR: {...}`
        - Or pure JSON objects with log data
        
        **Features:**
        - 📊 Statistics and visualizations
        - 🔍 Browse and filter log entries
        - 📤📥 View request/response pairs
        - 🚨 Error analysis and tracking
        - 📥 Export filtered results
        """)

### CSV Reviewer ###
with tab_csv:
    st.title("Enhanced CSV Reviewer")
    st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)
    
    csv_file = st.file_uploader(
        "Upload CSV (premise, hypothesis, label[, error_types, transformation_list])", type=["csv"], key="csv_uploader"
    )
    if csv_file:
        df = pd.read_csv(csv_file)
        has_err = 'error_types' in df.columns
        has_transformations = 'transformation_list' in df.columns
        
        # Sidebar for filtering and navigation
        st.sidebar.header("Filters & Navigation")
        
        if has_err:
            all_error_types = get_all_error_types(df)
            
            # Error type multi-select filter
            st.sidebar.subheader("Filter by Error Types")
            
            # Ensure default values exist in current options
            valid_defaults = [et for et in st.session_state.selected_error_types if et in all_error_types]
            
            selected_error_types = st.sidebar.multiselect(
                "Select error types to filter:",
                options=all_error_types,
                default=valid_defaults,
                key="error_type_filter"
            )
            st.session_state.selected_error_types = selected_error_types
            
            # Apply filters
            filtered_df = filter_by_error_types(df, selected_error_types)
            
            # Clear filters button
            if st.sidebar.button("Clear All Filters"):
                st.session_state.selected_error_types = []
                st.experimental_rerun()
        else:
            filtered_df = df
            st.sidebar.markdown("⚠️ No error_types column found")
        
        # Quick stats panel with dark theme styling
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", len(df))
        with col2:
            st.metric("Filtered Entries", len(filtered_df))
        with col3:
            if has_err and selected_error_types:
                st.metric("Selected Error Types", len(selected_error_types))
            else:
                st.metric("Unique Labels", df['label'].nunique())

        st.subheader("Data Preview")
        st.dataframe(filtered_df, use_container_width=True)

        # Interactive Error Types Analysis
        if has_err:
            st.subheader("Interactive Error Types Distribution")
            
            # Create error type counts
            error_counts = {}
            for error_str in df['error_types'].dropna():
                for error in parse_error_types(error_str):
                    error_counts[error] = error_counts.get(error, 0) + 1
            
            if error_counts:
                error_df = pd.DataFrame(list(error_counts.items()), columns=['Error Type', 'Count'])
                error_df = error_df.sort_values('Count', ascending=False)
                
                # Display error types as clickable buttons with dark theme
                st.markdown("**Click on error types below to filter dataset:**")
                
                # Create columns for error type buttons
                cols = st.columns(3)
                for i, (error_type, count) in enumerate(error_df.values):
                    col = cols[i % 3]
                    with col:
                        button_style = "🔹" if error_type not in st.session_state.selected_error_types else "✅"
                        if st.button(f"{button_style} {error_type} ({count})", key=f"error_btn_{error_type}"):
                            if error_type in st.session_state.selected_error_types:
                                st.session_state.selected_error_types.remove(error_type)
                            else:
                                st.session_state.selected_error_types.append(error_type)
                            st.experimental_rerun()
                
                # Show bar chart for visualization
                st.bar_chart(error_df.set_index('Error Type')['Count'])
                
                # Claim-Level Analysis for Selected Error Types
                if selected_error_types and has_transformations:
                    st.subheader(f"🔍 Claim-Level Analysis: {', '.join(selected_error_types)}")
                    
                    # Get all claims for selected error types
                    all_claims_for_error_types = []
                    for error_type in selected_error_types:
                        claims = get_claims_by_error_type(df, error_type)
                        all_claims_for_error_types.extend(claims)
                    
                    if all_claims_for_error_types:
                        st.markdown(f"**Found {len(all_claims_for_error_types)} claim transformations**")
                        
                        # Create a dataframe for claims
                        claims_df = pd.DataFrame(all_claims_for_error_types)
                        
                        # Display claims table
                        st.subheader("📋 All Claims with Selected Error Types")
                        
                        # Show claims in an expandable table
                        for i, claim in enumerate(all_claims_for_error_types):
                            with st.expander(f"Claim {i+1}: {claim['error_type']} (Doc: {claim['doc_id']})"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**🟢 Original Claim:**")
                                    st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 10px; border-radius: 5px; border: 1px solid #4CAF50;'>{claim['original_claim']}</div>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("**🔴 Transformed Claim:**")
                                    st.markdown(f"<div dir='rtl' style='background-color: #4d1a1a; color: white; padding: 10px; border-radius: 5px; border: 1px solid #f44336;'>{claim['transformed_claim']}</div>", unsafe_allow_html=True)
                                
                                # Show diff between original and transformed claim
                                st.markdown("**⚡ Transformation Diff:**")
                                diff_html = highlight_diff(claim['original_claim'], claim['transformed_claim'])
                                st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 10px; border-radius: 5px; border: 1px solid #666;'>{diff_html}</div>", unsafe_allow_html=True)
                                
                                # Show transformation details
                                st.markdown("**🔧 Transformation Details:**")
                                details = claim.get('transformation_details', {})
                                if 'diffs' in details:
                                    for diff in details['diffs']:
                                        st.markdown(f"<div style='color: #ffd700;'>• Changed '{diff.get('orig_text', '')}' → '{diff.get('trans_text', '')}' (positions {diff.get('orig_start', '')}-{diff.get('orig_end', '')})</div>", unsafe_allow_html=True)
                                elif 'original' in details and 'replacement' in details:
                                    st.markdown(f"<div style='color: #ffd700;'>• Replaced '{details['original']}' → '{details['replacement']}'</div>", unsafe_allow_html=True)
                                
                                # Quality rating for this specific claim transformation
                                rating_key = f"claim_quality_{i}_{claim['error_type']}_{hash(claim['transformed_claim'])}"
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    if st.button("👍 Good", key=f"good_{rating_key}"):
                                        st.session_state.quality_ratings[rating_key] = "good"
                                        st.success("✅")
                                
                                with col2:
                                    if st.button("👎 Poor", key=f"poor_{rating_key}"):
                                        st.session_state.quality_ratings[rating_key] = "poor"
                                        st.error("❌")
                                
                                with col3:
                                    if st.button("❓ Unclear", key=f"unclear_{rating_key}"):
                                        st.session_state.quality_ratings[rating_key] = "unclear"
                                        st.warning("❓")
                                
                                with col4:
                                    current_rating = st.session_state.quality_ratings.get(rating_key, "Not rated")
                                    st.markdown(f"<div style='color: #ffd700;'>Status: {current_rating}</div>", unsafe_allow_html=True)
                    
                    else:
                        st.markdown("⚠️ No transformation_list column found or no claims extracted")
                
                elif selected_error_types:
                    st.markdown("⚠️ transformation_list column not found. Cannot show claim-level analysis.")

        # Enhanced summary-level analysis (for documents with same doc_id)
        if 'doc_id' in df.columns:
            st.subheader("📄 Document-Level Summary Comparison")
            
            # Get unique doc_ids
            unique_docs = df['doc_id'].unique()
            selected_doc = st.selectbox("Select Document ID:", unique_docs, key="doc_selector")
            
            # Get all summaries for this doc_id
            doc_summaries = df[df['doc_id'] == selected_doc]
            
            if len(doc_summaries) > 1:
                st.markdown(f"**Found {len(doc_summaries)} summaries for document {selected_doc}**")
                
                # Find the gold/original summary (usually label='entailment' or no error_types)
                gold_summary = None
                corrupted_summaries = []
                
                for idx, row in doc_summaries.iterrows():
                    if row['label'] == 'entailment' or (has_err and pd.isna(row['error_types'])):
                        gold_summary = row['hypothesis']
                    else:
                        corrupted_summaries.append({
                            'hypothesis': row['hypothesis'],
                            'error_types': row.get('error_types', ''),
                            'label': row['label']
                        })
                
                if gold_summary and corrupted_summaries:
                    # Show gold summary
                    st.markdown("**🏆 Gold Summary:**")
                    st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #4CAF50;'>{gold_summary}</div>", unsafe_allow_html=True)
                    
                    # Show each corrupted summary with diff
                    st.markdown("**🔍 Corrupted Summaries vs Gold:**")
                    for i, corrupted in enumerate(corrupted_summaries):
                        with st.expander(f"Corrupted Summary {i+1}: {corrupted['error_types']} ({corrupted['label']})"):
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Corrupted Summary:**")
                                st.markdown(f"<div dir='rtl' style='background-color: #4d1a1a; color: white; padding: 10px; border-radius: 5px;'>{corrupted['hypothesis']}</div>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("**Diff vs Gold:**")
                                diff_html = highlight_diff(gold_summary, corrupted['hypothesis'])
                                st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 10px; border-radius: 5px;'>{diff_html}</div>", unsafe_allow_html=True)

        # Custom comparison (enhanced with dark theme)
        st.subheader("🔄 Custom Text Comparison")
        col1, col2 = st.columns(2)
        with col1:
            a = st.text_area("Original Text", key="ta", height=100)
        with col2:
            b = st.text_area("Corrupted Text", key="tb", height=100)
        
        if a and b:
            st.markdown("**Diff Visualization:**")
            st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 10px; border-radius: 5px;'>{highlight_diff(a, b)}</div>", unsafe_allow_html=True)

        # Enhanced row inspection with dark theme
        st.subheader("🔍 Inspect Individual Row")
        ridx = st.number_input("בחר שורה", 0, len(filtered_df)-1, 0, 1, key="ridx")
        if len(filtered_df) > 0:
            row = filtered_df.iloc[ridx]
            
            # Create expandable sections for better organization
            with st.expander("📄 Text Content", expanded=True):
                st.markdown(f"<div dir='rtl' style='color: white;'><strong>Premise:</strong><br/><div style='background-color: #1a4d1a; padding: 10px; border-radius: 5px; margin: 5px 0;'>{row['premise']}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div dir='rtl' style='color: white;'><strong>Hypothesis:</strong><br/><div style='background-color: #4d1a1a; padding: 10px; border-radius: 5px; margin: 5px 0;'>{row['hypothesis']}</div></div>", unsafe_allow_html=True)
                
                # Show claim-level transformations if available
                if has_transformations and pd.notna(row['transformation_list']):
                    claims = extract_claim_transformations(row['transformation_list'])
                    if claims:
                        st.markdown("<div style='color: white;'><strong>🔧 Claim-Level Transformations:</strong></div>", unsafe_allow_html=True)
                        for j, claim in enumerate(claims):
                            st.markdown(f"<div style='color: #ffd700; margin: 10px 0;'>Claim {j+1} ({claim['error_type']}):</div>", unsafe_allow_html=True)
                            diff_html = highlight_diff(claim['original_claim'], claim['transformed_claim'])
                            st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 8px; border-radius: 5px; margin: 5px 0;'>{diff_html}</div>", unsafe_allow_html=True)
            
            with st.expander("🏷️ Metadata"):
                st.markdown(f"<div style='color: white;'><strong>Label:</strong> {row['label']}</div>", unsafe_allow_html=True)
                if 'doc_id' in row:
                    st.markdown(f"<div style='color: white;'><strong>Doc ID:</strong> {row['doc_id']}</div>", unsafe_allow_html=True)
                if has_err:
                    st.markdown(f"<div style='color: white;'><strong>Error Types:</strong> {row['error_types']}</div>", unsafe_allow_html=True)
                    # Parse and display individual error types as badges
                    individual_errors = parse_error_types(row['error_types'])
                    if individual_errors:
                        st.markdown("<div style='color: white;'><strong>Individual Error Types:</strong></div>", unsafe_allow_html=True)
                        badge_html = ""
                        for error in individual_errors:
                            badge_html += f"<span style='background-color: #333; color: #ffd700; padding: 4px 12px; margin: 2px; border-radius: 12px; font-size: 0.9em; border: 1px solid #ffd700;'>{error}</span> "
                        st.markdown(badge_html, unsafe_allow_html=True)

        # Enhanced statistics section with dark theme
        st.subheader("📊 Dataset Statistics")
        
        # Label distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div style='color: white;'><strong>Label Distribution</strong></div>", unsafe_allow_html=True)
            label_counts = filtered_df['label'].value_counts()
            st.bar_chart(label_counts)
        
        with col2:
            if has_err and len(filtered_df) > 0:
                st.markdown("<div style='color: white;'><strong>Error Types in Filtered Data</strong></div>", unsafe_allow_html=True)
                filtered_error_counts = {}
                for error_str in filtered_df['error_types'].dropna():
                    for error in parse_error_types(error_str):
                        filtered_error_counts[error] = filtered_error_counts.get(error, 0) + 1
                
                if filtered_error_counts:
                    filtered_error_df = pd.DataFrame(list(filtered_error_counts.items()), columns=['Error Type', 'Count'])
                    filtered_error_df = filtered_error_df.sort_values('Count', ascending=False)
                    st.bar_chart(filtered_error_df.set_index('Error Type')['Count'])

        # Export functionality with dark theme
        if has_err and selected_error_types:
            st.subheader("📥 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export filtered data
                filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Filtered Data",
                    filtered_csv,
                    f"filtered_data_{'-'.join(selected_error_types[:2])}.csv",
                    "text/csv"
                )
            
            with col2:
                # Export quality ratings summary
                if st.session_state.quality_ratings:
                    ratings_df = pd.DataFrame([
                        {"example_id": k, "rating": v} 
                        for k, v in st.session_state.quality_ratings.items()
                    ])
                    ratings_csv = ratings_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 Download Quality Ratings",
                        ratings_csv,
                        "quality_ratings.csv",
                        "text/csv"
                    )

        # Word length analysis (enhanced with dark theme)
        st.subheader("📏 Text Length Analysis")
        if len(filtered_df) > 0:
            premise_lens = filtered_df['premise'].str.split().str.len()
            hyp_lens = filtered_df['hypothesis'].str.split().str.len()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Premise Length", f"{premise_lens.mean():.1f} words")
                st.metric("Premise Range", f"{premise_lens.min()}-{premise_lens.max()}")
            
            with col2:
                st.metric("Avg Hypothesis Length", f"{hyp_lens.mean():.1f} words")
                st.metric("Hypothesis Range", f"{hyp_lens.min()}-{hyp_lens.max()}")
            
            # Length distribution chart
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div style='color: white;'><strong>Premise Length Distribution</strong></div>", unsafe_allow_html=True)
                premise_hist = premise_lens.value_counts().sort_index()
                st.bar_chart(premise_hist)
            
            with col2:
                st.markdown("<div style='color: white;'><strong>Hypothesis Length Distribution</strong></div>", unsafe_allow_html=True)
                hyp_hist = hyp_lens.value_counts().sort_index()
                st.bar_chart(hyp_hist)

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

# Helper functions for dataset analysis (added after existing helpers)
def analyze_dataset_structure(data, file_format, filename):
    """Analyze dataset structure and extract key metrics"""
    analysis = {
        'filename': filename,
        'format': file_format,
        'total_records': 0,
        'error_distribution': {},
        'label_distribution': {},
        'source_distribution': {},
        'length_distribution': {},
        'doc_ids': set(),
        'columns': []
    }
    
    if file_format == 'json':
        # Handle JSON format (similar to existing JSON tab logic)
        analysis['total_records'] = len(data)
        all_doc_ids = set()
        error_counts = {}
        label_counts = {}
        source_counts = {}
        text_lengths = []
        
        for doc_idx, doc in enumerate(data):
            doc_id = doc.get('doc_id', f'doc_{doc_idx}')
            all_doc_ids.add(doc_id)
            
            # Extract source if available
            source = doc.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            
            # Analyze corrupted summaries
            for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
                # Error types
                for error_type in corrupted_summary.get('distinct_errors_type_names', []):
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                # Text length
                corrupted_text = corrupted_summary.get('corrupted_summary', '')
                if corrupted_text:
                    text_lengths.append(len(corrupted_text.split()))
            
            # Gold summary length
            gold_summary = doc.get('gold_summary', '')
            if gold_summary:
                text_lengths.append(len(gold_summary.split()))
        
        analysis['error_distribution'] = error_counts
        analysis['source_distribution'] = source_counts
        analysis['doc_ids'] = all_doc_ids
        analysis['length_distribution'] = {
            'mean': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'min': min(text_lengths) if text_lengths else 0,
            'max': max(text_lengths) if text_lengths else 0,
            'lengths': text_lengths
        }
        
    elif file_format == 'csv':
        # Handle CSV format (similar to existing CSV tab logic)
        analysis['total_records'] = len(data)
        analysis['columns'] = list(data.columns)
        
        # Doc IDs
        if 'doc_id' in data.columns:
            analysis['doc_ids'] = set(data['doc_id'].dropna().astype(str))
        
        # Error distribution
        if 'error_types' in data.columns:
            error_counts = {}
            for error_str in data['error_types'].dropna():
                for error in parse_error_types(error_str):
                    error_counts[error] = error_counts.get(error, 0) + 1
            analysis['error_distribution'] = error_counts
        
        # Label distribution
        if 'label' in data.columns:
            analysis['label_distribution'] = data['label'].value_counts().to_dict()
        
        # Source distribution
        if 'source' in data.columns:
            analysis['source_distribution'] = data['source'].value_counts().to_dict()
        
        # Text length analysis
        text_lengths = []
        if 'premise' in data.columns:
            premise_lengths = data['premise'].fillna('').str.split().str.len()
            text_lengths.extend(premise_lengths.dropna().tolist())
        if 'hypothesis' in data.columns:
            hyp_lengths = data['hypothesis'].fillna('').str.split().str.len()
            text_lengths.extend(hyp_lengths.dropna().tolist())
        
        if text_lengths:
            analysis['length_distribution'] = {
                'mean': sum(text_lengths) / len(text_lengths),
                'min': min(text_lengths),
                'max': max(text_lengths),
                'lengths': text_lengths
            }
    
    return analysis

def find_shared_doc_ids(analysis1, analysis2):
    """Find shared doc_ids between two datasets"""
    shared_ids = analysis1['doc_ids'].intersection(analysis2['doc_ids'])
    unique_to_1 = analysis1['doc_ids'] - analysis2['doc_ids']
    unique_to_2 = analysis2['doc_ids'] - analysis1['doc_ids']
    
    return {
        'shared_count': len(shared_ids),
        'shared_ids': sorted(list(shared_ids)),
        'unique_to_dataset1': len(unique_to_1),
        'unique_to_dataset2': len(unique_to_2),
        'overlap_percentage': len(shared_ids) / len(analysis1['doc_ids'].union(analysis2['doc_ids'])) * 100 if analysis1['doc_ids'].union(analysis2['doc_ids']) else 0
    }

def display_dataset_analysis(analysis, title_prefix=""):
    """Display analysis results for a single dataset"""
    st.subheader(f"📊 {title_prefix}Dataset Analysis: {analysis['filename']}")
    
    # Basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", analysis['total_records'])
    with col2:
        st.metric("Format", analysis['format'].upper())
    with col3:
        st.metric("Doc IDs", len(analysis['doc_ids']))
    
    # Error Type Distribution
    if analysis['error_distribution']:
        st.markdown("<div style='color: white;'><strong>🎯 Error Type Distribution</strong></div>", unsafe_allow_html=True)
        error_df = pd.DataFrame(list(analysis['error_distribution'].items()), columns=['Error Type', 'Count'])
        error_df = error_df.sort_values('Count', ascending=False)
        st.bar_chart(error_df.set_index('Error Type')['Count'])
        
        with st.expander("📋 Detailed Error Types"):
            st.dataframe(error_df, use_container_width=True)
    
    # Label Distribution
    if analysis['label_distribution']:
        st.markdown("<div style='color: white;'><strong>🏷️ Label Distribution</strong></div>", unsafe_allow_html=True)
        label_df = pd.DataFrame(list(analysis['label_distribution'].items()), columns=['Label', 'Count'])
        st.bar_chart(label_df.set_index('Label')['Count'])
    
    # Source Distribution
    if analysis['source_distribution']:
        st.markdown("<div style='color: white;'><strong>📰 Source Distribution</strong></div>", unsafe_allow_html=True)
        source_df = pd.DataFrame(list(analysis['source_distribution'].items()), columns=['Source', 'Count'])
        source_df = source_df.sort_values('Count', ascending=False)
        st.bar_chart(source_df.set_index('Source')['Count'])
    
    # Length Distribution
    if analysis['length_distribution'] and analysis['length_distribution']['lengths']:
        st.markdown("<div style='color: white;'><strong>📏 Text Length Distribution</strong></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Length", f"{analysis['length_distribution']['mean']:.1f} words")
        with col2:
            st.metric("Min Length", f"{analysis['length_distribution']['min']} words")
        with col3:
            st.metric("Max Length", f"{analysis['length_distribution']['max']} words")
        
        # Length histogram
        lengths = analysis['length_distribution']['lengths']
        length_series = pd.Series(lengths)
        length_counts = length_series.value_counts().sort_index()
        st.bar_chart(length_counts)

### Dataset Analysis ###
with tab_dataset:
    st.title("📊 Dataset Analysis & Comparison")
    st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)
    
    st.markdown("Upload one or two datasets (JSON/CSV) to analyze and compare their characteristics.")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div style='color: white;'><strong>📁 Dataset 1</strong></div>", unsafe_allow_html=True)
        file1 = st.file_uploader("Upload first dataset", type=["json", "csv"], key="dataset1")
    
    with col2:
        st.markdown("<div style='color: white;'><strong>📁 Dataset 2 (Optional)</strong></div>", unsafe_allow_html=True)
        file2 = st.file_uploader("Upload second dataset for comparison", type=["json", "csv"], key="dataset2")
    
    # Process uploaded files
    analysis1 = None
    analysis2 = None
    
    if file1:
        try:
            # Determine file format and load data
            file_format1 = "json" if file1.name.endswith('.json') else "csv"
            
            if file_format1 == "json":
                data1 = json.load(file1)
            else:
                data1 = pd.read_csv(file1)
            
            # Analyze dataset 1
            analysis1 = analyze_dataset_structure(data1, file_format1, file1.name)
            
        except Exception as e:
            st.error(f"❌ Error loading Dataset 1: {str(e)}")
    
    if file2:
        try:
            # Determine file format and load data
            file_format2 = "json" if file2.name.endswith('.json') else "csv"
            
            if file_format2 == "json":
                data2 = json.load(file2)
            else:
                data2 = pd.read_csv(file2)
            
            # Analyze dataset 2
            analysis2 = analyze_dataset_structure(data2, file_format2, file2.name)
            
        except Exception as e:
            st.error(f"❌ Error loading Dataset 2: {str(e)}")
    
    # Display analysis results
    if analysis1 and not analysis2:
        # Single dataset analysis
        display_dataset_analysis(analysis1)
        
        # Export option
        st.subheader("📥 Export Analysis")
        analysis_json = json.dumps({
            'analysis': analysis1,
            'shared_doc_ids': None
        }, default=str, ensure_ascii=False, indent=2).encode('utf-8')
        
        st.download_button(
            "📊 Download Analysis Report",
            analysis_json,
            f"dataset_analysis_{analysis1['filename']}.json",
            "application/json"
        )
    
    elif analysis1 and analysis2:
        # Comparative analysis
        st.header("🔄 Dataset Comparison")
        
        # Shared doc_ids analysis
        shared_analysis = find_shared_doc_ids(analysis1, analysis2)
        
        # Summary comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dataset 1 Records", analysis1['total_records'])
        with col2:
            st.metric("Dataset 2 Records", analysis2['total_records'])
        with col3:
            st.metric("Shared Doc IDs", shared_analysis['shared_count'])
        with col4:
            st.metric("Overlap %", f"{shared_analysis['overlap_percentage']:.1f}%")
        
        # Shared doc_ids details
        st.subheader("🔗 Shared Document IDs Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Shared Documents", shared_analysis['shared_count'])
        with col2:
            st.metric("Unique to Dataset 1", shared_analysis['unique_to_dataset1'])
        with col3:
            st.metric("Unique to Dataset 2", shared_analysis['unique_to_dataset2'])
        
        # Show exact shared doc_ids
        if shared_analysis['shared_ids']:
            with st.expander(f"📋 Exact Shared Doc IDs ({len(shared_analysis['shared_ids'])} documents)"):
                # Display as a formatted list
                shared_ids_text = ", ".join(shared_analysis['shared_ids'])
                st.text_area("Shared Doc IDs", shared_ids_text, height=200)
                
                # Also show as dataframe for better readability
                shared_df = pd.DataFrame({'Shared Doc IDs': shared_analysis['shared_ids']})
                st.dataframe(shared_df, use_container_width=True)
        else:
            st.markdown("<div style='color: white;'>⚠️ No shared doc_ids found between datasets</div>", unsafe_allow_html=True)
        
        # Side-by-side analysis
        st.header("📊 Side-by-Side Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_dataset_analysis(analysis1, "Dataset 1 - ")
        
        with col2:
            display_dataset_analysis(analysis2, "Dataset 2 - ")
        
        # Export comparison
        st.subheader("📥 Export Comparison")
        comparison_json = json.dumps({
            'dataset1_analysis': analysis1,
            'dataset2_analysis': analysis2,
            'shared_doc_ids_analysis': shared_analysis
        }, default=str, ensure_ascii=False, indent=2).encode('utf-8')
        
        st.download_button(
            "📊 Download Comparison Report",
            comparison_json,
            f"dataset_comparison_{analysis1['filename']}_vs_{analysis2['filename']}.json",
            "application/json"
        )
    
    elif not file1:
        # Instructions
        st.info("📁 Upload at least one dataset file to start analysis")
        st.markdown("""
        **Supported formats:**
        - **JSON**: Expected structure with `correctness_corrupted_summaries`, `doc_id`, `source`, etc.
        - **CSV**: Expected columns like `premise`, `hypothesis`, `label`, `error_types`, `doc_id`, `source`
        
        **Analysis features:**
        - 🎯 Error type distribution
        - 🏷️ Label distribution (entailment/contradiction)
        - 📰 Source distribution
        - 📏 Text length analysis
        - 🔗 Shared document ID analysis (when comparing 2 datasets)
        
        **Comparison features (2 datasets):**
        - Side-by-side metric comparison
        - Exact shared doc_id identification
        - Overlap percentage calculation
        - Export detailed comparison report
        """)