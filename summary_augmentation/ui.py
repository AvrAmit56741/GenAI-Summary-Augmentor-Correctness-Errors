import streamlit as st
import json
import pandas as pd
from difflib import SequenceMatcher
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
import zipfile
import io
from datetime import datetime
import csv


st.set_page_config(page_title="Hebrew NLI Analysis Hub", layout="wide", page_icon="🏠")


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

# Helper to count claims in text using SENTENCES method
def count_claims_sentences(text):
    """Count claims by splitting on sentence terminators (SENTENCES decomposition method)"""
    if not text or not isinstance(text, str):
        return 0
    # Split by common sentence terminators
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    return len(sentences)

# Helper to reduce corruption level in a summary
def reduce_corruption_level(corrupted_summary, target_percentage, strategy='random'):
    """
    Reduce corruption level by replacing some corrupted claims with originals.
    
    Args:
        corrupted_summary: Dict with corrupted_summary, transformation_list
        target_percentage: Target corruption percentage (0-100)
        strategy: 'random', 'max_transformation', or 'hebrew_focus'
    
    Returns:
        Modified corrupted_summary dict
    """
    import random
    import copy
    
    # Deep copy to avoid modifying original
    modified_summary = copy.deepcopy(corrupted_summary)
    transformation_list = modified_summary.get('transformation_list', [])
    
    if not transformation_list:
        return modified_summary
    
    # Count total claims
    corrupted_text = modified_summary.get('corrupted_summary', '')
    total_claims = count_claims_sentences(corrupted_text)
    
    if total_claims == 0:
        return modified_summary
    
    # Calculate target number of corrupted claims
    current_corruptions = len(transformation_list)
    target_corruptions = int(total_claims * target_percentage / 100)
    
    # Ensure at least 1 corruption if target_percentage > 0 and we have corruptions
    if target_percentage > 0 and target_corruptions == 0 and current_corruptions > 0:
        target_corruptions = 1
    
    # Can only reduce, not increase
    if target_corruptions >= current_corruptions:
        return modified_summary
    
    # Select which transformations to KEEP based on strategy
    if strategy == 'max_transformation':
        # Sort by number of error types (most errors first)
        sorted_transforms = sorted(
            enumerate(transformation_list),
            key=lambda x: len(x[1].get('applied_transformations', [])),
            reverse=True
        )
        indices_to_keep = [idx for idx, _ in sorted_transforms[:target_corruptions]]
    
    elif strategy == 'hebrew_focus':
        # Define Hebrew-specific error types
        hebrew_errors = {
            'hebrew_root_patten_confusion',
            'morphological_connective_confusion',
            'idiom_collocation_corruption',
            'noun_gender_swap',
            'verb_gender_swap',
            'verb_tense_swap',
            'directional_preposition_swap',
            'pronoun_reference_errors',
            'specificity_shift_errors',
            'construct_state_confusion',
            'homographic_gender_errors',
            'impersonal_to_personal_verb_errors',
            'incremental_temporal_state_mutations',
            'evidentiality_source_attribution_collapse',
            'verb_template_agency_reversal',
            'double_verb_dependency_errors',
            'subject_verb_person_mismatch'
        }
        
        # Calculate Hebrew error score for each transformation
        def calculate_hebrew_score(transform):
            applied_transformations = transform.get('applied_transformations', [])
            hebrew_count = sum(
                1 for t in applied_transformations 
                if t.get('type', '').lower() in hebrew_errors
            )
            return hebrew_count
        
        # Sort by Hebrew error count (highest first), then by total error count
        sorted_transforms = sorted(
            enumerate(transformation_list),
            key=lambda x: (
                calculate_hebrew_score(x[1]),  # Primary: Hebrew error count
                len(x[1].get('applied_transformations', []))  # Secondary: Total error count
            ),
            reverse=True
        )
        indices_to_keep = [idx for idx, _ in sorted_transforms[:target_corruptions]]
    
    else:  # random
        indices_to_keep = random.sample(range(len(transformation_list)), target_corruptions)
    
    # Build new transformation list (only kept transformations)
    new_transformation_list = [transformation_list[i] for i in sorted(indices_to_keep)]
    
    # Rebuild corrupted summary text
    # Split current corrupted summary into sentences
    sentences = [s.strip() for s in corrupted_text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    # Create mapping: for each transformation, find which sentence it corresponds to
    # Simple approach: match by transformed_claim text
    indices_to_revert = [i for i in range(len(transformation_list)) if i not in indices_to_keep]
    
    # Replace reverted claims with originals in the text
    new_summary_text = corrupted_text
    for idx in indices_to_revert:
        transform = transformation_list[idx]
        original = transform.get('original_claim', '')
        transformed = transform.get('transformed_claim', '')
        if transformed in new_summary_text:
            new_summary_text = new_summary_text.replace(transformed, original, 1)
    
    # Update the summary
    modified_summary['corrupted_summary'] = new_summary_text
    modified_summary['transformation_list'] = new_transformation_list
    modified_summary['transformations_performed'] = len(new_transformation_list)
    
    # Update distinct_errors_type_names to match new transformation_list
    new_error_types = set()
    for transform in new_transformation_list:
        for applied_trans in transform.get('applied_transformations', []):
            error_type = applied_trans.get('type', '')
            if error_type:
                new_error_types.add(error_type)
    modified_summary['distinct_errors_type_names'] = sorted(list(new_error_types))
    
    return modified_summary


def generate_metadata_text(modified_data, reduction_mode, config_params):
    """
    Generate metadata text file content with information about corruption reduction.
    
    Args:
        modified_data: The modified dataset
        reduction_mode: 'single' or 'plan'
        config_params: Dictionary with configuration parameters
            - For 'single': {'target_corruption', 'strategy', 'processed_count'}
            - For 'plan': {'parts_config', 'total_summaries'}
    
    Returns:
        String with formatted metadata
    """
    metadata_lines = []
    metadata_lines.append("=" * 80)
    metadata_lines.append("HEBREW SUMMARY CORRUPTION REDUCTION - METADATA")
    metadata_lines.append("=" * 80)
    metadata_lines.append("")
    
    # Generation timestamp
    metadata_lines.append(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    metadata_lines.append(f"Reduction Mode: {reduction_mode.upper()}")
    metadata_lines.append("")
    
    # Dataset name and basic info
    dataset_name = config_params.get('dataset_name', 'modified_dataset')
    metadata_lines.append(f"Dataset Name: {dataset_name}")
    metadata_lines.append(f"Total Documents: {len(modified_data)}")
    
    total_summaries = sum(len(doc.get('correctness_corrupted_summaries', [])) for doc in modified_data)
    metadata_lines.append(f"Total Summaries: {total_summaries}")
    metadata_lines.append("")
    
    # Reduction configuration
    metadata_lines.append("-" * 80)
    metadata_lines.append("REDUCTION CONFIGURATION")
    metadata_lines.append("-" * 80)
    metadata_lines.append("")
    
    if reduction_mode == 'single':
        metadata_lines.append("Mode: Single Reduction (uniform corruption level)")
        metadata_lines.append(f"Target Corruption Percentage: {config_params.get('target_corruption', 'N/A')}%")
        metadata_lines.append(f"Selection Strategy: {config_params.get('strategy', 'N/A')}")
        metadata_lines.append(f"Summaries Processed: {config_params.get('processed_count', 'N/A')}")
    else:  # plan
        metadata_lines.append("Mode: Plan Reduction (multiple corruption levels)")
        parts_config = config_params.get('parts_config', [])
        metadata_lines.append(f"Number of Parts: {len(parts_config)}")
        metadata_lines.append("")
        
        for i, part in enumerate(parts_config, 1):
            metadata_lines.append(f"  Part {i}:")
            metadata_lines.append(f"    - Samples: {part.get('samples', 0)}")
            metadata_lines.append(f"    - Target Corruption: {part.get('corruption', 0)}%")
            metadata_lines.append(f"    - Strategy: {part.get('strategy', 'N/A')}")
            metadata_lines.append("")
    
    # Statistical analysis
    metadata_lines.append("-" * 80)
    metadata_lines.append("CORRUPTION STATISTICS")
    metadata_lines.append("-" * 80)
    metadata_lines.append("")
    
    corruption_levels = []
    corruption_percentages = []
    error_type_counts = {}
    
    for doc in modified_data:
        for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
            transformation_list = corrupted_summary.get('transformation_list', [])
            actual_count = len(transformation_list)
            corrupted_text = corrupted_summary.get('corrupted_summary', '')
            
            # Count claims
            total_claims = len([s.strip() for s in corrupted_text.replace('!', '.').replace('?', '.').split('.') if s.strip()])
            corruption_percentage = (actual_count / total_claims * 100) if total_claims > 0 else 0.0
            
            corruption_levels.append(actual_count)
            corruption_percentages.append(corruption_percentage)
            
            # Count error types
            for transform in transformation_list:
                for applied_trans in transform.get('applied_transformations', []):
                    error_type = applied_trans.get('type', 'unknown')
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
    
    if corruption_levels:
        metadata_lines.append(f"Average Corruptions per Summary: {sum(corruption_levels) / len(corruption_levels):.2f}")
        metadata_lines.append(f"Min Corruptions: {min(corruption_levels)}")
        metadata_lines.append(f"Max Corruptions: {max(corruption_levels)}")
        metadata_lines.append(f"Median Corruptions: {sorted(corruption_levels)[len(corruption_levels)//2]}")
        metadata_lines.append("")
        
        metadata_lines.append(f"Average Corruption Percentage: {sum(corruption_percentages) / len(corruption_percentages):.2f}%")
        metadata_lines.append(f"Min Corruption Percentage: {min(corruption_percentages):.2f}%")
        metadata_lines.append(f"Max Corruption Percentage: {max(corruption_percentages):.2f}%")
        metadata_lines.append("")
    
    # Error type distribution
    metadata_lines.append("-" * 80)
    metadata_lines.append("ERROR TYPE DISTRIBUTION")
    metadata_lines.append("-" * 80)
    metadata_lines.append("")
    
    if error_type_counts:
        total_errors = sum(error_type_counts.values())
        metadata_lines.append(f"Total Error Instances: {total_errors}")
        metadata_lines.append(f"Unique Error Types: {len(error_type_counts)}")
        metadata_lines.append("")
        metadata_lines.append("Error Type Breakdown:")
        
        for error_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            metadata_lines.append(f"  - {error_type}: {count} ({percentage:.2f}%)")
    else:
        metadata_lines.append("No errors in dataset (all claims are original)")
    
    metadata_lines.append("")
    metadata_lines.append("=" * 80)
    metadata_lines.append("END OF METADATA")
    metadata_lines.append("=" * 80)
    
    return "\n".join(metadata_lines)


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
tab_json, tab_csv, tab_utils, tab_analysis, tab_report, tab_logs, tab_dataset, tab_modification, tab_llm_parse, tab_corruption_level = st.tabs([
    "JSON Analyzer", "CSV Reviewer", "Utils", "Analysis", "Error Analyzer", "API Logs", "Dataset Analysis", "Dataset Modification", "LLM Parse Data", "Corrupted Level"
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
        
        # Corruption Level Statistics
        st.subheader("📊 Corruption Level Statistics")
        
        # Calculate corruption statistics for all summaries
        all_corruption_levels = []
        all_corruption_percentages = []
        corruption_details = []
        inconsistencies_found = 0
        
        for doc_idx, doc in enumerate(data):
            for summary_idx, corrupted_summary in enumerate(doc.get('correctness_corrupted_summaries', [])):
                # Manual count from transformation_list (GROUND TRUTH)
                transformation_list = corrupted_summary.get('transformation_list', [])
                actual_count = len(transformation_list)
                
                # Stored value (may be inconsistent)
                stored_count = corrupted_summary.get('transformations_performed', 0)
                
                # Count total claims in corrupted summary using SENTENCES method
                corrupted_text = corrupted_summary.get('corrupted_summary', '')
                total_claims = count_claims_sentences(corrupted_text)
                
                # Calculate corruption percentage
                corruption_percentage = (actual_count / total_claims * 100) if total_claims > 0 else 0.0
                
                # Check for inconsistency
                is_inconsistent = (actual_count != stored_count)
                if is_inconsistent:
                    inconsistencies_found += 1
                
                all_corruption_levels.append(actual_count)
                all_corruption_percentages.append(corruption_percentage)
                corruption_details.append({
                    'doc_idx': doc_idx,
                    'summary_idx': summary_idx,
                    'actual_transformations': actual_count,
                    'stored_transformations': stored_count,
                    'total_claims': total_claims,
                    'corruption_percentage': f"{corruption_percentage:.1f}%",
                    'inconsistent': '⚠️' if is_inconsistent else '✓'
                })
        
        if all_corruption_levels:
            avg_corruption = sum(all_corruption_levels) / len(all_corruption_levels)
            min_corruption = min(all_corruption_levels)
            max_corruption = max(all_corruption_levels)
            avg_corruption_pct = sum(all_corruption_percentages) / len(all_corruption_percentages)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Average Corruptions", f"{avg_corruption:.2f}")
            with col2:
                st.metric("Min Corruptions", min_corruption)
            with col3:
                st.metric("Max Corruptions", max_corruption)
            with col4:
                st.metric("Total Summaries", len(all_corruption_levels))
            with col5:
                st.metric("Avg Corruption %", f"{avg_corruption_pct:.1f}%")
            with col6:
                inconsistency_label = "Inconsistencies ⚠️" if inconsistencies_found > 0 else "Data Quality ✓"
                st.metric(inconsistency_label, inconsistencies_found)
            
            # Show distribution chart
            st.markdown("<div style='color: white;'><strong>Corruption Level Distribution:</strong></div>", unsafe_allow_html=True)
            corruption_df = pd.DataFrame({'Transformations Performed': all_corruption_levels})
            st.bar_chart(corruption_df['Transformations Performed'].value_counts().sort_index())
            
            # Show detailed table in expander
            with st.expander("📋 View Detailed Corruption Stats per Summary"):
                details_df = pd.DataFrame(corruption_details)
                details_df = details_df.sort_values('actual_transformations', ascending=False)
                
                # Show warning if inconsistencies found
                if inconsistencies_found > 0:
                    st.warning(f"⚠️ Found {inconsistencies_found} inconsistencies between stored and actual counts. Using ACTUAL counts from transformation_list.")
                else:
                    st.success("✅ All stored counts match actual transformation_list lengths.")
                
                st.dataframe(details_df, use_container_width=True)
                
                # Sample selector for reviewing article and summary
                st.markdown("---")
                st.markdown("<div style='color: white;'><strong>🔍 Review Sample Details:</strong></div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_doc_idx = st.selectbox(
                        "Select Document Index:",
                        options=sorted(details_df['doc_idx'].unique()),
                        key="corruption_stats_doc_selector"
                    )
                with col2:
                    # Filter summaries for selected document
                    available_summaries = details_df[details_df['doc_idx'] == selected_doc_idx]['summary_idx'].tolist()
                    selected_summary_idx = st.selectbox(
                        "Select Summary Index:",
                        options=available_summaries,
                        key="corruption_stats_summary_selector"
                    )
                
                # Display selected sample details
                if selected_doc_idx is not None and selected_summary_idx is not None:
                    selected_doc = data[selected_doc_idx]
                    selected_summary = selected_doc.get('correctness_corrupted_summaries', [])[selected_summary_idx]
                    
                    # Manual count (GROUND TRUTH)
                    actual_count = len(selected_summary.get('transformation_list', []))
                    stored_count = selected_summary.get('transformations_performed', 0)
                    
                    # Count total claims using SENTENCES method
                    corrupted_text = selected_summary.get('corrupted_summary', '')
                    total_claims = count_claims_sentences(corrupted_text)
                    corruption_pct = (actual_count / total_claims * 100) if total_claims > 0 else 0.0
                    
                    # Show metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Actual Claims Corrupted", f"{actual_count}")
                    with col2:
                        st.metric("Stored Value", f"{stored_count}")
                    with col3:
                        st.metric("Total Claims", f"{total_claims}")
                    with col4:
                        st.metric("Corruption %", f"{corruption_pct:.1f}%")
                    with col5:
                        match_status = "✅ Match" if actual_count == stored_count else "⚠️ Mismatch"
                        st.metric("Status", match_status)
                    
                    # Show source article
                    st.markdown("<div style='color: white;'><strong>📰 Source Article:</strong></div>", unsafe_allow_html=True)
                    st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 15px; border-radius: 5px; max-height: 300px; overflow-y: auto;'>{selected_doc.get('source_article', '')}</div>", unsafe_allow_html=True)
                    
                    # Show gold summary
                    st.markdown("<div style='color: white;'><strong>🏆 Gold Summary:</strong></div>", unsafe_allow_html=True)
                    st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #4CAF50;'>{selected_doc.get('gold_summary', '')}</div>", unsafe_allow_html=True)
                    
                    # Show corrupted summary
                    st.markdown("<div style='color: white;'><strong>🔴 Corrupted Summary:</strong></div>", unsafe_allow_html=True)
                    st.markdown(f"<div dir='rtl' style='background-color: #4d1a1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #f44336;'>{selected_summary.get('corrupted_summary', '')}</div>", unsafe_allow_html=True)
                    
                    # Show diff
                    st.markdown("<div style='color: white;'><strong>⚡ Difference Visualization:</strong></div>", unsafe_allow_html=True)
                    diff_html = highlight_diff(selected_doc.get('gold_summary', ''), selected_summary.get('corrupted_summary', ''))
                    st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 15px; border-radius: 5px; border: 1px solid #666;'>{diff_html}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No corruption statistics available")
        
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
                        # Download JSON of all displayed claim transformations (without source_article, gold_summary)
                        sanitized_claims = [
                            {k: v for k, v in claim.items() if k not in ("source_article", "gold_summary")}
                            for claim in all_claims_for_error_types
                        ]
                        export_payload = json.dumps(sanitized_claims, ensure_ascii=False, indent=2).encode('utf-8')
                        st.download_button(
                            "📥 Download Claims JSON",
                            export_payload,
                            f"claims_{'_'.join(selected_error_types[:2])}.json",
                            "application/json",
                            help="Download all visible claim-level transformations for the selected error types"
                        )
                        
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

        # Additional distribution statistics
        col1, col2 = st.columns(2)
        
        # Source distribution
        with col1:
            if 'source' in filtered_df.columns and len(filtered_df) > 0:
                st.markdown("<div style='color: white;'><strong>Source Distribution</strong></div>", unsafe_allow_html=True)
                source_counts = filtered_df['source'].value_counts()
                st.bar_chart(source_counts)
        
        # Length group distribution
        with col2:
            if 'len_group' in filtered_df.columns and len(filtered_df) > 0:
                st.markdown("<div style='color: white;'><strong>Length Group Distribution</strong></div>", unsafe_allow_html=True)
                length_group_counts = filtered_df['len_group'].value_counts()
                st.bar_chart(length_group_counts)

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
    st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)
    
    # Combine CSV
    st.subheader("📁 Combine CSV Files")
    
    # Mode selection
    csv_combine_mode = st.radio("Combine Mode:", ["Default", "Custom Sample Size"], key="csv_mode", horizontal=True)
    
    csvs = st.file_uploader("CSV to combine", type=["csv"], accept_multiple_files=True, key="ccomb")
    if csvs:
        try:
            # Custom sample size configuration
            file_sample_limits = {}
            file_label_limits = {}
            if csv_combine_mode == "Custom Sample Size":
                st.markdown("<div style='color: white;'><strong>🎛️ Configure Sample Sizes:</strong></div>", unsafe_allow_html=True)
                
                # Option to enable per-label sampling
                per_label_sampling = st.checkbox("Enable per-label sampling", key="per_label_csv", help="Sample specific amounts per label (entailment/contradiction)")
                
                for i, f in enumerate(csvs):
                    df_temp = pd.read_csv(f)
                    max_samples = len(df_temp)
                    f.seek(0)  # Reset file pointer
                    
                    if per_label_sampling and 'label' in df_temp.columns:
                        # Per-label sampling
                        st.markdown(f"<div style='color: #ffd700;'><strong>📁 {f.name}</strong></div>", unsafe_allow_html=True)
                        
                        label_counts = df_temp['label'].value_counts()
                        max_entailment = label_counts.get('entailment', 0)
                        max_contradiction = label_counts.get('contradiction', 0)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            ent_size = st.number_input(
                                f"Entailment (max: {max_entailment})",
                                min_value=0,
                                max_value=max_entailment,
                                value=max_entailment,
                                key=f"csv_ent_{i}"
                            )
                        with col2:
                            con_size = st.number_input(
                                f"Contradiction (max: {max_contradiction})",
                                min_value=0,
                                max_value=max_contradiction,
                                value=max_contradiction,
                                key=f"csv_con_{i}"
                            )
                        
                        file_label_limits[f.name] = {'entailment': ent_size, 'contradiction': con_size}
                    else:
                        # Total sampling
                        sample_size = st.number_input(
                            f"{f.name} (max: {max_samples})",
                            min_value=1,
                            max_value=max_samples,
                            value=max_samples,
                            key=f"csv_sample_{i}"
                        )
                        file_sample_limits[f.name] = sample_size
            
            # Combine CSV files
            combined_dfs = []
            file_info = []
            
            for f in csvs:
                df_temp = pd.read_csv(f)
                original_count = len(df_temp)
                
                if csv_combine_mode == "Custom Sample Size":
                    # Check if per-label sampling is configured
                    if f.name in file_label_limits:
                        # Apply per-label sampling
                        label_limits = file_label_limits[f.name]
                        sampled_dfs = []
                        
                        for label, limit in label_limits.items():
                            if limit > 0:
                                label_df = df_temp[df_temp['label'] == label].head(limit)
                                sampled_dfs.append(label_df)
                        
                        df_temp = pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else pd.DataFrame()
                        file_info.append(f"{f.name}: {len(df_temp)}/{original_count} samples (per-label)")
                    else:
                        # Apply total sampling
                        limit = file_sample_limits.get(f.name, original_count)
                        df_temp = df_temp.head(limit)
                        file_info.append(f"{f.name}: {len(df_temp)}/{original_count} samples")
                else:
                    file_info.append(f"{f.name}: {original_count} samples")
                
                combined_dfs.append(df_temp)
            
            dfc = pd.concat(combined_dfs, ignore_index=True)
            
            # Debug information
            st.markdown("<div style='color: white;'><strong>🔍 Debug Information:</strong></div>", unsafe_allow_html=True)
            debug_cols = st.columns(len(combined_dfs))
            for i, (df, col) in enumerate(zip(combined_dfs, debug_cols)):
                with col:
                    st.metric(f"File {i+1} Contributed", len(df))
            
            # Check for duplicates
            col1, col2, col3 = st.columns(3)
            with col1:
                duplicates = dfc.duplicated().sum()
                st.metric("Duplicate Rows", duplicates)
            with col2:
                if 'doc_id' in dfc.columns:
                    unique_docs = dfc['doc_id'].nunique()
                    st.metric("Unique Doc IDs", unique_docs)
                else:
                    st.metric("Expected Total", sum(len(df) for df in combined_dfs))
            with col3:
                st.metric("Actual Total", len(dfc))
            
            # Data preview
            st.markdown("<div style='color: white;'><strong>📊 Combined Data Preview:</strong></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(dfc))
            with col2:
                st.metric("Total Columns", len(dfc.columns))
            with col3:
                if 'label' in dfc.columns:
                    st.metric("Unique Labels", dfc['label'].nunique())
                else:
                    st.metric("Files Combined", len(csvs))
            
            # Show sample data
            with st.expander("🔍 Sample Data (First 5 rows)"):
                st.dataframe(dfc.head(), use_container_width=True)
            
            # Label distribution if available
            if 'label' in dfc.columns:
                st.markdown("<div style='color: white;'><strong>🏷️ Label Distribution:</strong></div>", unsafe_allow_html=True)
                label_counts = dfc['label'].value_counts()
                st.bar_chart(label_counts)
                
                # Label-specific length statistics
                if 'premise' in dfc.columns and 'hypothesis' in dfc.columns:
                    st.markdown("<div style='color: white;'><strong>📏 Label-Specific Length Statistics:</strong></div>", unsafe_allow_html=True)
                    
                    # Calculate lengths
                    entailment_df = dfc[dfc['label'] == 'entailment']
                    contradiction_df = dfc[dfc['label'] == 'contradiction']
                    
                    # Premise lengths
                    premise_ent_len = entailment_df['premise'].str.split().str.len().mean() if len(entailment_df) > 0 else 0
                    premise_con_len = contradiction_df['premise'].str.split().str.len().mean() if len(contradiction_df) > 0 else 0
                    
                    # Hypothesis lengths
                    hyp_ent_len = entailment_df['hypothesis'].str.split().str.len().mean() if len(entailment_df) > 0 else 0
                    hyp_con_len = contradiction_df['hypothesis'].str.split().str.len().mean() if len(contradiction_df) > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Premise Avg (Entailment)", f"{premise_ent_len:.1f} words")
                    with col2:
                        st.metric("Premise Avg (Contradiction)", f"{premise_con_len:.1f} words")
                    with col3:
                        st.metric("Hypothesis Avg (Entailment)", f"{hyp_ent_len:.1f} words")
                    with col4:
                        st.metric("Hypothesis Avg (Contradiction)", f"{hyp_con_len:.1f} words")
            
            # Create summary file
            from datetime import datetime
            import zipfile
            import io
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_text = f"""CSV Combination Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Mode: {csv_combine_mode}
Total Files Combined: {len(csvs)}
Total Records: {len(dfc)}

Files Combined:
"""
            for info in file_info:
                summary_text += f"  - {info}\n"
            
            # Create ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add combined CSV
                zip_file.writestr("combined.csv", dfc.to_csv(index=False))
                # Add summary file
                zip_file.writestr("combination_summary.txt", summary_text)
            
            zip_buffer.seek(0)
            st.download_button(
                "📦 Download Combined CSV + Summary (ZIP)",
                zip_buffer.getvalue(),
                f"combined_csv_{timestamp}.zip",
                "application/zip"
            )
            
        except Exception as e:
            st.error(f"❌ Error combining CSV files: {str(e)}")
    
    # Combine JSON
    st.subheader("🗂️ Combine JSON Files")
    
    # Mode selection
    json_combine_mode = st.radio("Combine Mode:", ["Default", "Custom Sample Size"], key="json_mode", horizontal=True)
    
    jsns = st.file_uploader("JSON to combine", type=["json"], accept_multiple_files=True, key="jcomb")
    if jsns:
        # Custom sample size configuration
        file_sample_limits_json = {}
        if json_combine_mode == "Custom Sample Size":
            st.markdown("<div style='color: white;'><strong>🎛️ Configure Sample Sizes:</strong></div>", unsafe_allow_html=True)
            for i, f in enumerate(jsns):
                try:
                    data_temp = json.load(f)
                    f.seek(0)  # Reset file pointer
                    
                    if isinstance(data_temp, list):
                        max_samples = len(data_temp)
                    else:
                        max_samples = 1
                    
                    sample_size = st.number_input(
                        f"{f.name} (max: {max_samples})",
                        min_value=1,
                        max_value=max_samples,
                        value=max_samples,
                        key=f"json_sample_{i}"
                    )
                    file_sample_limits_json[f.name] = sample_size
                except:
                    pass
        
        lst = []
        errors = []
        file_info = []
        
        for f in jsns:
            try:
                data = json.load(f)
                original_count = 0
                
                if isinstance(data, list):
                    original_count = len(data)
                    
                    if json_combine_mode == "Custom Sample Size":
                        limit = file_sample_limits_json.get(f.name, original_count)
                        data = data[:limit]
                        file_info.append(f"{f.name}: {len(data)}/{original_count} records")
                    else:
                        file_info.append(f"{f.name}: {original_count} records")
                    
                    lst.extend(data)
                else:
                    original_count = 1
                    file_info.append(f"{f.name}: 1 record")
                    lst.append(data)
                    
            except json.JSONDecodeError as e:
                errors.append(f"❌ {f.name}: {str(e)}")
            except Exception as e:
                errors.append(f"❌ {f.name}: {str(e)}")
        
        # Show errors if any
        if errors:
            st.warning(f"⚠️ Skipped {len(errors)} file(s) with errors:")
            for error in errors:
                st.error(error)
        
        # Show results if we have valid data
        if lst:
            # Data preview
            st.markdown("<div style='color: white;'><strong>📊 Combined JSON Preview:</strong></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(lst))
            with col2:
                st.metric("Files Combined", f"{len(jsns) - len(errors)}/{len(jsns)}")

            # Show sample structure
            with st.expander("🔍 Sample Structure (First record)"):
                st.json(lst[0] if len(lst) > 0 else {})
            
            # Create summary file
            from datetime import datetime
            import zipfile
            import io
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_text = f"""JSON Combination Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Mode: {json_combine_mode}
Total Files Combined: {len(jsns) - len(errors)}
Total Records: {len(lst)}

Files Combined:
"""
            for info in file_info:
                summary_text += f"  - {info}\n"
            
            if errors:
                summary_text += f"\nSkipped Files ({len(errors)}):\n"
                for error in errors:
                    summary_text += f"  - {error}\n"
            
            # Create ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add combined JSON
                zip_file.writestr("combined.json", json.dumps(lst, ensure_ascii=False, indent=2))
                # Add summary file
                zip_file.writestr("combination_summary.txt", summary_text)
            
            zip_buffer.seek(0)
            st.download_button(
                "📦 Download Combined JSON + Summary (ZIP)",
                zip_buffer.getvalue(),
                f"combined_json_{timestamp}.zip",
                "application/zip",
                key="download_combined_json"
            )
        elif not errors:
            st.warning("⚠️ No valid JSON data found in uploaded files")
    
    # Convert JSON Augmentation to CSV (Claim-Level)
    st.subheader("🔄 Convert JSON Augmentation to CSV (Claim-Level)")
    st.markdown("<div style='color: white;'>Convert augmented JSON files to CSV format with claim-level labels for model training.</div>", unsafe_allow_html=True)
    
    # Mode selection
    conversion_mode = st.radio(
        "Conversion Mode:",
        ["Balance", "Contradiction-only"],
        key="claim_conversion_mode",
        horizontal=True,
        help="Balance: Creates paired samples (entailment + contradiction). Contradiction-only: Only corrupted claims with errors."
    )
    
    # Sample limit
    max_samples = st.number_input(
        "Maximum contradiction samples to convert:",
        min_value=1,
        value=1000,
        step=100,
        key="claim_max_samples",
        help="Number of contradiction (transformed) samples. In Balance mode, entailment samples will be added based on the ratio."
    )
    
    # Ratio configuration for Balance mode
    contradiction_per_entailment = 1
    if conversion_mode == "Balance":
        col1, col2 = st.columns([2, 1])
        with col1:
            contradiction_per_entailment = st.number_input(
                "For every N contradictions, add 1 entailment:",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key="contradiction_per_entailment",
                help="Example: 5 = For every 5 contradictions, add 1 entailment. 1 = Add 1 entailment for each contradiction (50/50 balance)"
            )
        with col2:
            entailment_count_estimate = max_samples // contradiction_per_entailment
            st.metric("Est. Entailments", f"~{entailment_count_estimate}", 
                     help=f"Estimated entailment samples to be added")
    
    json_to_csv_file = st.file_uploader("Upload JSON augmentation file", type=["json"], key="json_to_csv")
    
    if json_to_csv_file:
        try:
            json_data = json.load(json_to_csv_file)
            
            # Convert to list if single object
            if isinstance(json_data, dict):
                json_data = [json_data]
            
            # Extract claim-level data
            contradiction_rows = []
            entailment_candidates = []
            sample_count = 0
            
            # First, collect all contradiction samples (up to max_samples)
            for doc in json_data:
                if sample_count >= max_samples:
                    break
                    
                source = doc.get('source_article', '')
                doc_id = doc.get('doc_id', '')
                
                for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
                    if sample_count >= max_samples:
                        break
                        
                    for transformation in corrupted_summary.get('transformation_list', []):
                        if sample_count >= max_samples:
                            break
                            
                        # Extract error types for this specific claim
                        error_types = ','.join([
                            t['type'] for t in transformation.get('applied_transformations', [])
                        ])
                        
                        # Add contradiction sample (corrupted claim)
                        contradiction_rows.append({
                            'premise': source,
                            'hypothesis': transformation.get('transformed_claim', ''),
                            'error_types': error_types,
                            'label': 'contradiction',
                            'doc_id': doc_id,
                            'original_claim': transformation.get('original_claim', '')
                        })
                        
                        # Store entailment candidate for later
                        if conversion_mode == "Balance":
                            entailment_candidates.append({
                                'premise': source,
                                'hypothesis': transformation.get('original_claim', ''),
                                'error_types': '',
                                'label': 'entailment',
                                'doc_id': doc_id,
                                'original_claim': transformation.get('original_claim', '')
                            })
                        
                        sample_count += 1
            
            # Now add entailments based on ratio
            rows = contradiction_rows.copy()
            if conversion_mode == "Balance" and entailment_candidates:
                num_entailments_to_add = len(contradiction_rows) // contradiction_per_entailment
                # Add entailments (up to available candidates)
                for i in range(min(num_entailments_to_add, len(entailment_candidates))):
                    rows.append(entailment_candidates[i])
            
            if rows:
                df_converted = pd.DataFrame(rows)
                
                # Clean data: remove null bytes and problematic characters
                for col in df_converted.columns:
                    if df_converted[col].dtype == 'object':
                        df_converted[col] = df_converted[col].fillna('').astype(str).str.replace('\x00', '', regex=False)
                
                # Show preview
                st.markdown("<div style='color: white;'><strong>📊 Conversion Preview:</strong></div>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", len(df_converted))
                with col2:
                    st.metric("Contradictions", len(df_converted[df_converted['label'] == 'contradiction']))
                with col3:
                    st.metric("Entailments", len(df_converted[df_converted['label'] == 'entailment']))
                with col4:
                    st.metric("Unique Documents", df_converted['doc_id'].nunique())
                
                # Show error type distribution
                st.markdown("<div style='color: white;'><strong>🎯 Error Type Distribution:</strong></div>", unsafe_allow_html=True)
                error_type_counts = {}
                for error_str in df_converted['error_types'].dropna():
                    for error in error_str.split(','):
                        error = error.strip()
                        if error:
                            error_type_counts[error] = error_type_counts.get(error, 0) + 1
                
                if error_type_counts:
                    error_dist_df = pd.DataFrame(list(error_type_counts.items()), columns=['Error Type', 'Count'])
                    error_dist_df = error_dist_df.sort_values('Count', ascending=False)
                    st.bar_chart(error_dist_df.set_index('Error Type')['Count'])
                
                # Show sample data
                with st.expander("🔍 Sample Data (First 5 rows)"):
                    st.dataframe(df_converted.head(), use_container_width=True)
                
                # Download button
                csv_buffer = io.StringIO()
                df_converted.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_NONNUMERIC)
                csv_data = csv_buffer.getvalue().encode('utf-8')
                
                st.download_button(
                    label="📥 Download Claim-Level CSV",
                    data=csv_data,
                    file_name="augmentation_claim_level.csv",
                    mime="text/csv",
                    key="download_json_to_csv",
                    help="Download CSV with claim-level labels ready for model training"
                )
                
                if conversion_mode == "Balance":
                    num_contradictions = len(df_converted[df_converted['label'] == 'contradiction'])
                    num_entailments = len(df_converted[df_converted['label'] == 'entailment'])
                    st.success(f"✅ Successfully converted {sample_count} transformations into {len(df_converted)} balanced samples ({num_contradictions} contradictions + {num_entailments} entailments)!")
                else:
                    st.success(f"✅ Successfully converted {len(df_converted)} contradiction samples from JSON augmentation!")
            else:
                st.warning("⚠️ No claim transformations found in the JSON file")
        
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON file: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error converting JSON to CSV: {str(e)}")
    else:
        st.info("📁 Upload a JSON augmentation file to convert to claim-level CSV format")
        st.markdown("""
        **Expected JSON structure:**
        - Root array of documents with `source_article`, `doc_id`
        - Each document contains `correctness_corrupted_summaries`
        - Each summary contains `transformation_list` with claim transformations
        - Each transformation has `original_claim`, `transformed_claim`, and `applied_transformations`
        
        **Conversion Modes:**
        
        **Balance Mode:**
        - Includes ALL contradiction samples (transformed claims with errors)
        - Adds entailment samples based on configurable ratio
        - Ratio examples:
          - N=1: For every 1 contradiction, add 1 entailment (50/50 balance)
          - N=5: For every 5 contradictions, add 1 entailment (83% contradiction / 17% entailment)
          - N=2: For every 2 contradictions, add 1 entailment (67% contradiction / 33% entailment)
        - `label`: 'entailment' for original claims, 'contradiction' for corrupted claims
        - `error_types`: Empty for entailment, populated for contradiction
        
        **Contradiction-only Mode:**
        - Only outputs corrupted claims with errors
        - Output: 100% contradiction samples
        - CSV rows = number of transformations
        - `label`: Always 'contradiction'
        - `error_types`: Always populated with error types
        
        **Output CSV columns:**
        - `premise`: Source article text
        - `hypothesis`: Claim text (original or transformed)
        - `error_types`: Comma-separated error types (empty for entailment)
        - `label`: 'entailment' or 'contradiction'
        - `doc_id`: Document identifier
        - `original_claim`: Original claim before transformation
        
        **Use case:** Training multi-label error type classifiers on claim-level data
        """)
    
    # Convert between JSON and CSV formats
    st.subheader("🔄 Convert Between JSON ↔ CSV Formats")
    st.markdown("<div style='color: white;'>Convert between JSON augmentation format and CSV format (summary-level).</div>", unsafe_allow_html=True)
    
    conversion_direction = st.radio("Conversion Direction:", ["JSON → CSV", "CSV → JSON"], key="conversion_direction", horizontal=True)
    
    if conversion_direction == "JSON → CSV":
        # JSON to CSV conversion
        json_convert_file = st.file_uploader("Upload JSON file to convert", type=["json"], key="json_convert_file")
        
        if json_convert_file:
            try:
                json_data = json.load(json_convert_file)
                
                # Convert to list if single object
                if isinstance(json_data, dict):
                    json_data = [json_data]
                
                # Extract summary-level data
                rows = []
                for doc in json_data:
                    source = doc.get('source_article', '')
                    doc_id = doc.get('doc_id', '')
                    gold_summary = doc.get('gold_summary', '')
                    
                    # Add entailment row (gold summary)
                    rows.append({
                        'premise': source,
                        'hypothesis': gold_summary,
                        'label': 'entailment',
                        'error_types': '',
                        'doc_id': doc_id
                    })
                    
                    # Add contradiction rows (corrupted summaries)
                    for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
                        error_types = ','.join(corrupted_summary.get('distinct_errors_type_names', []))
                        
                        rows.append({
                            'premise': source,
                            'hypothesis': corrupted_summary.get('corrupted_summary', ''),
                            'label': 'contradiction',
                            'error_types': error_types,
                            'doc_id': doc_id
                        })
                
                if rows:
                    df_converted = pd.DataFrame(rows)
                    
                    # Show preview
                    st.markdown("<div style='color: white;'><strong>📊 Conversion Preview:</strong></div>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(df_converted))
                    with col2:
                        st.metric("Entailment", len(df_converted[df_converted['label'] == 'entailment']))
                    with col3:
                        st.metric("Contradiction", len(df_converted[df_converted['label'] == 'contradiction']))
                    
                    with st.expander("🔍 Sample Data (First 5 rows)"):
                        st.dataframe(df_converted.head(), use_container_width=True)
                    
                    # Download button
                    csv_data = df_converted.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC).encode('utf-8')
                    st.download_button(
                        "📥 Download CSV",
                        csv_data,
                        "converted_from_json.csv",
                        "text/csv",
                        key="download_json_to_csv"
                    )
                    
                    st.success(f"✅ Successfully converted to CSV with {len(df_converted)} rows!")
                else:
                    st.warning("⚠️ No data found in the JSON file")
            
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON file: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error converting JSON to CSV: {str(e)}")
    
    else:  # CSV → JSON
        csv_convert_file = st.file_uploader("Upload CSV file to convert", type=["csv"], key="csv_convert_file")
        
        if csv_convert_file:
            try:
                df_csv = pd.read_csv(csv_convert_file)
                
                # Validate required columns
                required_cols = ['premise', 'hypothesis', 'label', 'doc_id']
                missing_cols = [col for col in required_cols if col not in df_csv.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Group by doc_id
                    json_data = []
                    
                    for doc_id, group in df_csv.groupby('doc_id'):
                        # Find entailment row (gold summary)
                        entailment_rows = group[group['label'] == 'entailment']
                        
                        if len(entailment_rows) > 0:
                            gold_row = entailment_rows.iloc[0]
                            source_article = gold_row['premise']
                            gold_summary = gold_row['hypothesis']
                        else:
                            # No entailment, use first row's premise
                            source_article = group.iloc[0]['premise']
                            gold_summary = ""
                        
                        # Get contradiction rows (corrupted summaries)
                        contradiction_rows = group[group['label'] == 'contradiction']
                        corrupted_summaries = []
                        
                        for _, row in contradiction_rows.iterrows():
                            error_types = []
                            if 'error_types' in row and pd.notna(row['error_types']) and row['error_types']:
                                error_types = [e.strip() for e in str(row['error_types']).split(',') if e.strip()]
                            
                            corrupted_summaries.append({
                                'corrupted_summary': row['hypothesis'],
                                'distinct_errors_type_names': error_types,
                                'transformation_list': [],
                                'transformations_performed': len(error_types),
                                'transformation_score': 0.0
                            })
                        
                        json_data.append({
                            'doc_id': str(doc_id),
                            'source_article': source_article,
                            'gold_summary': gold_summary,
                            'correctness_corrupted_summaries': corrupted_summaries
                        })
                    
                    if json_data:
                        # Show preview
                        st.markdown("<div style='color: white;'><strong>📊 Conversion Preview:</strong></div>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Documents", len(json_data))
                        with col2:
                            total_corrupted = sum(len(doc['correctness_corrupted_summaries']) for doc in json_data)
                            st.metric("Corrupted Summaries", total_corrupted)
                        with col3:
                            st.metric("Original CSV Rows", len(df_csv))
                        
                        with st.expander("🔍 Sample Structure (First document)"):
                            st.json(json_data[0] if len(json_data) > 0 else {})
                        
                        # Download button
                        json_output = json.dumps(json_data, ensure_ascii=False, indent=2).encode('utf-8')
                        st.download_button(
                            "📥 Download JSON",
                            json_output,
                            "converted_from_csv.json",
                            "application/json",
                            key="download_csv_to_json"
                        )
                        
                        st.success(f"✅ Successfully converted to JSON with {len(json_data)} documents!")
                    else:
                        st.warning("⚠️ No data to convert")
            
            except Exception as e:
                st.error(f"❌ Error converting CSV to JSON: {str(e)}")
    
    # Split Data - Enhanced with independent data loading
    st.subheader("📊 Split Data")
    
    # Allow independent CSV upload for splitting
    split_csv = st.file_uploader("Upload CSV to split", type=["csv"], key="split_csv")
    
    # Use either uploaded CSV or existing df from CSV tab
    df_to_split = None
    if split_csv:
        try:
            df_to_split = pd.read_csv(split_csv)
            st.success(f"✅ Loaded {len(df_to_split)} records from uploaded file")
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
    elif 'df' in locals():
        df_to_split = df.copy()
        st.info("ℹ️ Using data from CSV Reviewer tab")
    
    if df_to_split is not None:
        # Validate required columns
        if 'label' not in df_to_split.columns:
            st.error("❌ CSV must contain 'label' column for stratified splitting")
        else:
            # Show original dataset stats
            st.markdown("<div style='color: white;'><strong>📈 Original Dataset Statistics:</strong></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df_to_split))
            with col2:
                label_dist = df_to_split['label'].value_counts()
                st.metric("Entailment", label_dist.get('entailment', 0))
            with col3:
                st.metric("Contradiction", label_dist.get('contradiction', 0))
            
            split_mode = st.selectbox("Split Mode", ["kfold", "naive"], key="mode")
            
        if split_mode == 'kfold':
                st.markdown("<div style='color: white;'><strong>🔄 K-Fold Cross Validation</strong></div>", unsafe_allow_html=True)
                k = st.number_input("Number of folds (K)", 2, 20, 5, key="k")
                
                if st.button("🚀 Generate K-Fold Splits", key="generate_kfold"):
                    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                    
                    # Store split data for analysis
                    split_data = {}
                    
                    # Also store in session state for ZIP download
                    st.session_state.kfold_split_data = {}
                    st.session_state.kfold_k_value = k
                    
                    st.markdown("<div style='color: white;'><strong>📥 Download Split Files:</strong></div>", unsafe_allow_html=True)
                    
                    # Generate all splits and collect statistics
                    for i, (train_idx, test_idx) in enumerate(skf.split(df_to_split, df_to_split['label'])):
                        train_df = df_to_split.iloc[train_idx].copy()
                        test_df = df_to_split.iloc[test_idx].copy()
                        
                        # Store for analysis
                        split_data[f'train_{i}'] = train_df
                        split_data[f'test_{i}'] = test_df
                        
                        # Also store in session state for ZIP download
                        st.session_state.kfold_split_data[f'train_{i}'] = train_df
                        st.session_state.kfold_split_data[f'test_{i}'] = test_df
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                f"📥 Train Fold {i+1}", 
                                train_df.to_csv(index=False).encode('utf-8'), 
                                f"train_fold_{i+1}.csv",
                                "text/csv",
                                key=f"train_{i}"
                            )
                        with col2:
                            st.download_button(
                                f"📥 Test Fold {i+1}", 
                                test_df.to_csv(index=False).encode('utf-8'), 
                                f"test_fold_{i+1}.csv",
                                "text/csv",
                                key=f"test_{i}"
                            )
                    
                    # Comprehensive statistics analysis
                    st.markdown("<div style='color: white;'><strong>📊 Split Quality Analysis:</strong></div>", unsafe_allow_html=True)
                    
                    # Create statistics table
                    stats_data = []
                    for split_name, split_df in split_data.items():
                        # Basic stats
                        total_records = len(split_df)
                        entailment_count = len(split_df[split_df['label'] == 'entailment'])
                        contradiction_count = len(split_df[split_df['label'] == 'contradiction'])
                        entailment_pct = (entailment_count / total_records * 100) if total_records > 0 else 0
                        
                        # Error types analysis
                        error_types_info = "N/A"
                        if 'error_types' in split_df.columns:
                            error_counts = {}
                            for error_str in split_df['error_types'].dropna():
                                for error in parse_error_types(error_str):
                                    error_counts[error] = error_counts.get(error, 0) + 1
                            if error_counts:
                                all_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
                                error_types_info = ", ".join([f"{err}: {cnt}" for err, cnt in all_errors])
                        
                        # Text length analysis
                        premise_lengths = split_df['premise'].str.split().str.len() if 'premise' in split_df.columns else pd.Series([])
                        hypothesis_lengths = split_df['hypothesis'].str.split().str.len() if 'hypothesis' in split_df.columns else pd.Series([])
                        
                        avg_premise_len = premise_lengths.mean() if not premise_lengths.empty else 0
                        avg_hypothesis_len = hypothesis_lengths.mean() if not hypothesis_lengths.empty else 0
                        
                        stats_data.append({
                            'Split': split_name,
                            'Total Records': total_records,
                            'Entailment': entailment_count,
                            'Contradiction': contradiction_count,
                            'Entailment %': f"{entailment_pct:.1f}%",
                            'All Error Types': error_types_info,
                            'Avg Premise Length': f"{avg_premise_len:.1f}" if avg_premise_len > 0 else "N/A",
                            'Avg Hypothesis Length': f"{avg_hypothesis_len:.1f}" if avg_hypothesis_len > 0 else "N/A"
                        })
                    
                    # Display statistics table
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Error Type Mapping Table
                    st.markdown("<div style='color: white;'><strong>📋 Error Type Mapping:</strong></div>", unsafe_allow_html=True)
                    
                    # Collect all error types across all splits for mapping table
                    all_error_types_mapping = {}
                    for split_name, split_df in split_data.items():
                        if 'error_types' in split_df.columns:
                            error_counts = {}
                            for error_str in split_df['error_types'].dropna():
                                for error in parse_error_types(error_str):
                                    error_counts[error] = error_counts.get(error, 0) + 1
                            all_error_types_mapping[split_name] = error_counts
                    
                    if all_error_types_mapping:
                        # Get all unique error types
                        all_unique_errors = set()
                        for error_counts in all_error_types_mapping.values():
                            all_unique_errors.update(error_counts.keys())
                        
                        # Create mapping table data
                        mapping_data = []
                        for error_type in sorted(all_unique_errors):
                            row = {'Error Type': error_type}
                            total_count = 0
                            
                            # Add counts for each split
                            for split_name in sorted(all_error_types_mapping.keys()):
                                count = all_error_types_mapping[split_name].get(error_type, 0)
                                row[split_name] = count
                                total_count += count
                            
                            # Calculate ratios
                            ratio_parts = []
                            for split_name in sorted(all_error_types_mapping.keys()):
                                count = all_error_types_mapping[split_name].get(error_type, 0)
                                ratio = (count / total_count * 100) if total_count > 0 else 0
                                ratio_parts.append(f"{ratio:.1f}%")
                            
                            row['Ratio'] = " : ".join(ratio_parts)
                            mapping_data.append(row)
                        
                        # Display mapping table
                        mapping_df = pd.DataFrame(mapping_data)
                        st.dataframe(mapping_df, use_container_width=True)
                    
                    # Error Types Distribution Comparison
                    st.markdown("<div style='color: white;'><strong>🎯 Error Types Distribution Across Splits:</strong></div>", unsafe_allow_html=True)
                    
                    # Collect all error types across all splits
                    all_error_types_data = {}
                    for split_name, split_df in split_data.items():
                        if 'error_types' in split_df.columns:
                            error_counts = {}
                            for error_str in split_df['error_types'].dropna():
                                for error in parse_error_types(error_str):
                                    error_counts[error] = error_counts.get(error, 0) + 1
                            all_error_types_data[split_name] = error_counts
                    
                    if all_error_types_data:
                        # Create a comprehensive DataFrame for error types
                        all_unique_errors = set()
                        for error_counts in all_error_types_data.values():
                            all_unique_errors.update(error_counts.keys())
                        
                        error_comparison_data = {}
                        for split_name in all_error_types_data.keys():
                            error_comparison_data[split_name] = {}
                            for error_type in all_unique_errors:
                                error_comparison_data[split_name][error_type] = all_error_types_data[split_name].get(error_type, 0)
                        
                        error_comparison_df = pd.DataFrame(error_comparison_data).T
                        error_comparison_df = error_comparison_df.fillna(0)
                        
                        # Display the error types comparison chart
                        st.bar_chart(error_comparison_df)
                    
                    # Visualization of label distribution across splits
                    st.markdown("<div style='color: white;'><strong>📈 Label Distribution Across Splits:</strong></div>", unsafe_allow_html=True)
                    
                    # Prepare data for visualization
                    viz_data = {}
                    for split_name, split_df in split_data.items():
                        label_counts = split_df['label'].value_counts()
                        viz_data[split_name] = {
                            'entailment': label_counts.get('entailment', 0),
                            'contradiction': label_counts.get('contradiction', 0)
                        }
                    
                    # Create comparison chart
                    viz_df = pd.DataFrame(viz_data).T
                    st.bar_chart(viz_df)
                    
                    # Text length comparison
                    if 'premise' in df_to_split.columns and 'hypothesis' in df_to_split.columns:
                        st.markdown("<div style='color: white;'><strong>📏 Text Length Comparison:</strong></div>", unsafe_allow_html=True)
                        
                        length_data = {}
                        for split_name, split_df in split_data.items():
                            premise_lens = split_df['premise'].str.split().str.len()
                            hypothesis_lens = split_df['hypothesis'].str.split().str.len()
                            length_data[split_name] = {
                                'Avg Premise Length': premise_lens.mean(),
                                'Avg Hypothesis Length': hypothesis_lens.mean()
                            }
                        
                        length_df = pd.DataFrame(length_data).T
                        st.bar_chart(length_df)
                    
                # Download all splits as ZIP (outside the generate button)
                if 'kfold_split_data' in st.session_state and st.session_state.kfold_split_data:
                    st.subheader("📦 Download All Splits")
                    
                    import zipfile
                    import io
                    
                    # Create ZIP file in memory
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for split_name, split_df in st.session_state.kfold_split_data.items():
                            csv_data = split_df.to_csv(index=False)
                            zip_file.writestr(f"{split_name}.csv", csv_data)
                    
                    zip_buffer.seek(0)
                    k_value = st.session_state.get('kfold_k_value', 'unknown')
                    st.download_button(
                        "📦 Download All K-Fold Splits as ZIP",
                        zip_buffer.getvalue(),
                        f"kfold_splits_{k_value}folds.zip",
                        "application/zip",
                        help=f"Download all {len(st.session_state.kfold_split_data)} split files in one ZIP package"
                    )
            
                else:  # naive mode
                    st.markdown("<div style='color: white;'><strong>📈 Train/Validation/Test Split</strong></div>", unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        pval = st.slider("Validation %", 0.0, 0.5, 0.1, key="pval")
                    with col2:
                        ptr = st.slider("Test %", 0.0, 0.5, 0.1, key="ptr")
                
                train_pct = 1.0 - pval - ptr
                st.markdown(f"<div style='color: white;'>**Split Ratios:** Train: {train_pct:.1%}, Validation: {pval:.1%}, Test: {ptr:.1%}</div>", unsafe_allow_html=True)
                
                if train_pct <= 0:
                    st.error("❌ Train percentage must be positive. Reduce validation and test percentages.")
                elif st.button("🚀 Generate Train/Val/Test Splits", key="generate_naive"):
                    try:
                        # First split: separate train from (val + test)
                        train_df, temp_df = train_test_split(
                            df_to_split, 
                            test_size=(pval + ptr), 
                            stratify=df_to_split['label'], 
                            random_state=42
                        )
                        
                        # Second split: separate val from test
                        if pval > 0 and ptr > 0:
                            val_df, test_df = train_test_split(
                                temp_df, 
                                test_size=ptr/(pval + ptr), 
                                stratify=temp_df['label'], 
                                random_state=42
                            )
                        elif pval > 0:
                            val_df = temp_df
                            test_df = pd.DataFrame(columns=df_to_split.columns)
                        else:
                            val_df = pd.DataFrame(columns=df_to_split.columns)
                            test_df = temp_df
                        
                        # Store for analysis
                        split_data = {
                            'train': train_df,
                            'validation': val_df,
                            'test': test_df
                        }
                        
                        # Also store in session state for ZIP download
                        st.session_state.naive_split_data = {
                            'train': train_df,
                            'validation': val_df,
                            'test': test_df
                        }
                        
                        # Download buttons
                        st.markdown("<div style='color: white;'><strong>📥 Download Split Files:</strong></div>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                "📥 Train Set", 
                                train_df.to_csv(index=False).encode('utf-8'), 
                                "train.csv",
                                "text/csv"
                            )
                        with col2:
                            if len(val_df) > 0:
                                st.download_button(
                                    "📥 Validation Set", 
                                    val_df.to_csv(index=False).encode('utf-8'), 
                                    "validation.csv",
                                    "text/csv"
                                )
                        with col3:
                            if len(test_df) > 0:
                                st.download_button(
                                    "📥 Test Set", 
                                    test_df.to_csv(index=False).encode('utf-8'), 
                                    "test.csv",
                                    "text/csv"
                                )
                        
                        # Comprehensive statistics analysis
                        st.markdown("<div style='color: white;'><strong>📊 Split Quality Analysis:</strong></div>", unsafe_allow_html=True)
                        
                        # Create statistics table
                        stats_data = []
                        for split_name, split_df in split_data.items():
                            if len(split_df) == 0:
                                continue
                                
                            # Basic stats
                            total_records = len(split_df)
                            entailment_count = len(split_df[split_df['label'] == 'entailment'])
                            contradiction_count = len(split_df[split_df['label'] == 'contradiction'])
                            entailment_pct = (entailment_count / total_records * 100) if total_records > 0 else 0
                            
                            # Error types analysis
                            error_types_info = "N/A"
                            if 'error_types' in split_df.columns:
                                error_counts = {}
                                for error_str in split_df['error_types'].dropna():
                                    for error in parse_error_types(error_str):
                                        error_counts[error] = error_counts.get(error, 0) + 1
                                if error_counts:
                                    all_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
                                    error_types_info = ", ".join([f"{err}: {cnt}" for err, cnt in all_errors])
                            
                            # Text length analysis
                            premise_lengths = split_df['premise'].str.split().str.len() if 'premise' in split_df.columns else pd.Series([])
                            hypothesis_lengths = split_df['hypothesis'].str.split().str.len() if 'hypothesis' in split_df.columns else pd.Series([])
                            
                            avg_premise_len = premise_lengths.mean() if not premise_lengths.empty else 0
                            avg_hypothesis_len = hypothesis_lengths.mean() if not hypothesis_lengths.empty else 0
                            
                            stats_data.append({
                                'Split': split_name.capitalize(),
                                'Total Records': total_records,
                                'Entailment': entailment_count,
                                'Contradiction': contradiction_count,
                                'Entailment %': f"{entailment_pct:.1f}%",
                                'All Error Types': error_types_info,
                                'Avg Premise Length': f"{avg_premise_len:.1f}" if avg_premise_len > 0 else "N/A",
                                'Avg Hypothesis Length': f"{avg_hypothesis_len:.1f}" if avg_hypothesis_len > 0 else "N/A"
                            })
                        
                        # Display statistics table
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Error Type Mapping Table
                        st.markdown("<div style='color: white;'><strong>📋 Error Type Mapping:</strong></div>", unsafe_allow_html=True)
                        
                        # Collect all error types across all splits for mapping table
                        all_error_types_mapping = {}
                        for split_name, split_df in split_data.items():
                            if len(split_df) > 0 and 'error_types' in split_df.columns:
                                error_counts = {}
                                for error_str in split_df['error_types'].dropna():
                                    for error in parse_error_types(error_str):
                                        error_counts[error] = error_counts.get(error, 0) + 1
                                all_error_types_mapping[split_name.capitalize()] = error_counts
                        
                        if all_error_types_mapping:
                            # Get all unique error types
                            all_unique_errors = set()
                            for error_counts in all_error_types_mapping.values():
                                all_unique_errors.update(error_counts.keys())
                            
                            # Create mapping table data
                            mapping_data = []
                            for error_type in sorted(all_unique_errors):
                                row = {'Error Type': error_type}
                                total_count = 0
                                
                                # Add counts for each split
                                for split_name in sorted(all_error_types_mapping.keys()):
                                    count = all_error_types_mapping[split_name].get(error_type, 0)
                                    row[split_name] = count
                                    total_count += count
                                
                                # Calculate ratios
                                ratio_parts = []
                                for split_name in sorted(all_error_types_mapping.keys()):
                                    count = all_error_types_mapping[split_name].get(error_type, 0)
                                    ratio = (count / total_count * 100) if total_count > 0 else 0
                                    ratio_parts.append(f"{ratio:.1f}%")
                                
                                row['Ratio'] = " : ".join(ratio_parts)
                                mapping_data.append(row)
                            
                            # Display mapping table
                            mapping_df = pd.DataFrame(mapping_data)
                            st.dataframe(mapping_df, use_container_width=True)
                        
                        # Error Types Distribution Comparison
                        st.markdown("<div style='color: white;'><strong>🎯 Error Types Distribution Across Splits:</strong></div>", unsafe_allow_html=True)
                        
                        # Collect all error types across all splits
                        all_error_types_data = {}
                        for split_name, split_df in split_data.items():
                            if len(split_df) > 0 and 'error_types' in split_df.columns:
                                error_counts = {}
                                for error_str in split_df['error_types'].dropna():
                                    for error in parse_error_types(error_str):
                                        error_counts[error] = error_counts.get(error, 0) + 1
                                all_error_types_data[split_name.capitalize()] = error_counts
                        
                        if all_error_types_data:
                            # Create a comprehensive DataFrame for error types
                            all_unique_errors = set()
                            for error_counts in all_error_types_data.values():
                                all_unique_errors.update(error_counts.keys())
                            
                            error_comparison_data = {}
                            for split_name in all_error_types_data.keys():
                                error_comparison_data[split_name] = {}
                                for error_type in all_unique_errors:
                                    error_comparison_data[split_name][error_type] = all_error_types_data[split_name].get(error_type, 0)
                            
                            error_comparison_df = pd.DataFrame(error_comparison_data).T
                            error_comparison_df = error_comparison_df.fillna(0)
                            
                            # Display the error types comparison chart
                            st.bar_chart(error_comparison_df)
                        
                        # Visualization of label distribution across splits
                        st.markdown("<div style='color: white;'><strong>📈 Label Distribution Across Splits:</strong></div>", unsafe_allow_html=True)
                        
                        # Prepare data for visualization
                        viz_data = {}
                        for split_name, split_df in split_data.items():
                            if len(split_df) > 0:
                                label_counts = split_df['label'].value_counts()
                                viz_data[split_name.capitalize()] = {
                                    'entailment': label_counts.get('entailment', 0),
                                    'contradiction': label_counts.get('contradiction', 0)
                                }
                        
                        # Create comparison chart
                        if viz_data:
                            viz_df = pd.DataFrame(viz_data).T
                            st.bar_chart(viz_df)
                        
                        # Text length comparison
                        if 'premise' in df_to_split.columns and 'hypothesis' in df_to_split.columns:
                            st.markdown("<div style='color: white;'><strong>📏 Text Length Comparison:</strong></div>", unsafe_allow_html=True)
                            
                            length_data = {}
                            for split_name, split_df in split_data.items():
                                if len(split_df) > 0:
                                    premise_lens = split_df['premise'].str.split().str.len()
                                    hypothesis_lens = split_df['hypothesis'].str.split().str.len()
                                    length_data[split_name.capitalize()] = {
                                        'Avg Premise Length': premise_lens.mean(),
                                        'Avg Hypothesis Length': hypothesis_lens.mean()
                                    }
                            
                            if length_data:
                                length_df = pd.DataFrame(length_data).T
                                st.bar_chart(length_df)
                        
                        # Download all splits as ZIP (outside the generate button)
                        if 'naive_split_data' in st.session_state and st.session_state.naive_split_data:
                            st.subheader("📦 Download All Splits")
                            
                            import zipfile
                            import io
                            
                            # Create ZIP file in memory
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for split_name, split_df in st.session_state.naive_split_data.items():
                                    if len(split_df) > 0:
                                        csv_data = split_df.to_csv(index=False)
                                        zip_file.writestr(f"{split_name}.csv", csv_data)
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                "📦 Download All Train/Val/Test Splits as ZIP",
                                zip_buffer.getvalue(),
                                f"train_val_test_splits.zip",
                                "application/zip",
                                help=f"Download all split files in one ZIP package"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error creating splits: {str(e)}")
    else:
        st.info("📁 Upload a CSV file or load data in the CSV Reviewer tab to enable splitting functionality")

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
        st.markdown(f"**severity**  {rec['severity']}")
        st.markdown(f"**Doc_ID:**  {rec['doc_id']}")
        st.markdown(f"**Premise:**  {rec['premise']}")
        st.markdown(f"**AI Summary:**  {rec['ai_summary']}")
        st.markdown(f"**Generative_summary:**  {rec['generative_summary']}")
        st.markdown(f"**Error Name:**  {rec['correctness_error_name']}")
        st.markdown(f"**Reasoning:**  {rec['correctness_error_reasoning']}")
        st.markdown(f"**Features:**  {rec['mrl_unique_hebrew_features']}")
        st.markdown(f"**Why_its_factual_error:**  {rec['why_its_factual_error']}")
        st.markdown(f"**Evidence (Summary):**  {rec['evidence_from_summary']}")
        st.markdown(f"**Evidence (Source):**  {rec['evidence_from_source_article']}")
        #st.markdown(f"**severity**  {rec['severity']}")

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

### Dataset Modification ###
with tab_modification:
    st.title("🔧 Dataset Modification")
    st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)
    
    st.markdown("Filter your CSV dataset based on categorical feature values with SQL-like functionality.")
    
    # File upload section
    st.subheader("📁 Data Loading")
    modification_csv = st.file_uploader("Upload CSV to modify", type=["csv"], key="modification_csv")
    
    # Use either uploaded CSV or existing df from CSV tab
    df_to_modify = None
    if modification_csv:
        try:
            df_to_modify = pd.read_csv(modification_csv)
            st.success(f"✅ Loaded {len(df_to_modify)} records from uploaded file")
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
    elif 'df' in locals():
        df_to_modify = df.copy()
        st.info("ℹ️ Using data from CSV Reviewer tab")
    
    if df_to_modify is not None:
        # Initialize session state for filter configuration
        if 'filter_config' not in st.session_state:
            st.session_state.filter_config = {
                'mode': 'Filter Out',
                'selected_columns': [],
                'column_values': {}
            }
        
        # Show original dataset stats
        st.subheader("📊 Original Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df_to_modify))
        with col2:
            st.metric("Total Columns", len(df_to_modify.columns))
        with col3:
            # Get categorical columns (non-numeric)
            categorical_cols = []
            for col in df_to_modify.columns:
                if df_to_modify[col].dtype == 'object' or df_to_modify[col].dtype.name == 'category':
                    categorical_cols.append(col)
            st.metric("Categorical Columns", len(categorical_cols))
        
        if not categorical_cols:
            st.error("❌ No categorical columns found in the dataset. Please upload a CSV with categorical features.")
        else:
            # Filter configuration section
            st.subheader("🎛️ Filter Configuration")
            
            # Filter mode selection
            filter_mode = st.radio(
                "Filter Mode:",
                ["Filter Out", "Filter In"],
                index=0,  # Filter Out is default
                help="Filter Out: Remove rows matching criteria | Filter In: Keep only rows matching criteria"
            )
            st.session_state.filter_config['mode'] = filter_mode
            
            # Column selection
            st.markdown("<div style='color: white;'><strong>Select Columns to Filter On:</strong></div>", unsafe_allow_html=True)
            selected_columns = st.multiselect(
                "Choose categorical columns:",
                options=categorical_cols,
                default=st.session_state.filter_config['selected_columns'],
                key="filter_columns"
            )
            st.session_state.filter_config['selected_columns'] = selected_columns
            
            if selected_columns:
                # Value selection for each column
                st.markdown("<div style='color: white;'><strong>Select Values for Each Column:</strong></div>", unsafe_allow_html=True)
                
                column_filters = {}
                for col in selected_columns:
                    with st.expander(f"📋 {col} - Value Selection"):
                        # Get unique values and their counts
                        value_counts = df_to_modify[col].value_counts()
                        
                        # Create options with counts
                        options_with_counts = []
                        for value, count in value_counts.items():
                            if pd.notna(value):  # Skip NaN values
                                options_with_counts.append(f"{value} ({count})")
                        
                        # Multi-select for values
                        selected_values_with_counts = st.multiselect(
                            f"Select values to {filter_mode.lower()}:",
                            options=options_with_counts,
                            key=f"values_{col}"
                        )
                        
                        # Extract actual values (remove counts)
                        selected_values = []
                        for item in selected_values_with_counts:
                            actual_value = item.split(' (')[0]
                            selected_values.append(actual_value)
                        
                        column_filters[col] = selected_values
                        
                        if selected_values:
                            st.markdown(f"<div style='color: #ffd700;'>Selected: {', '.join(selected_values)}</div>", unsafe_allow_html=True)
                
                st.session_state.filter_config['column_values'] = column_filters
                
                # Filter preview section
                if any(values for values in column_filters.values()):
                    st.subheader("📊 Filter Preview")
                    
                    # Build filter summary
                    filter_conditions = []
                    for col, values in column_filters.items():
                        if values:
                            if len(values) == 1:
                                filter_conditions.append(f"{col} = '{values[0]}'")
                            else:
                                values_str = "', '".join(values)
                                filter_conditions.append(f"{col} IN ('{values_str}')")
                    
                    if filter_conditions:
                        filter_summary = " AND ".join(filter_conditions)
                        mode_text = "Remove" if filter_mode == "Filter Out" else "Keep only"
                        
                        st.markdown("<div style='color: white;'><strong>Filter Summary:</strong></div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='color: #ffd700; background-color: #2d2d2d; padding: 10px; border-radius: 5px; border: 1px solid #666;'>{mode_text} rows where: {filter_summary}</div>", unsafe_allow_html=True)
                        
                        # Calculate preview counts
                        try:
                            # Build pandas query
                            query_conditions = []
                            for col, values in column_filters.items():
                                if values:
                                    if len(values) == 1:
                                        query_conditions.append(f"`{col}` == '{values[0]}'")
                                    else:
                                        values_list = "', '".join(values)
                                        query_conditions.append(f"`{col}` in ['{values_list}']")
                            
                            if query_conditions:
                                query_string = " and ".join(query_conditions)
                                matching_rows = df_to_modify.query(query_string)
                                matching_count = len(matching_rows)
                                
                                if filter_mode == "Filter Out":
                                    final_count = len(df_to_modify) - matching_count
                                    preview_text = f"Will remove {matching_count} rows, keeping {final_count} rows"
                                else:
                                    final_count = matching_count
                                    preview_text = f"Will keep {final_count} rows, removing {len(df_to_modify) - matching_count} rows"
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Original Records", len(df_to_modify))
                                with col2:
                                    st.metric("Matching Criteria", matching_count)
                                with col3:
                                    st.metric("Final Records", final_count)
                                
                                st.markdown(f"<div style='color: white;'><strong>{preview_text}</strong></div>", unsafe_allow_html=True)
                                
                                # Apply filters button
                                if st.button("🚀 Apply Filters", key="apply_filters"):
                                    try:
                                        if filter_mode == "Filter Out":
                                            # Remove matching rows
                                            filtered_df = df_to_modify.query(f"not ({query_string})")
                                        else:
                                            # Keep only matching rows
                                            filtered_df = matching_rows.copy()
                                        
                                        # Store filtered data in session state
                                        st.session_state.filtered_data = filtered_df
                                        st.success(f"✅ Filter applied successfully! Dataset now has {len(filtered_df)} records.")
                                        
                                    except Exception as e:
                                        st.error(f"❌ Error applying filters: {str(e)}")
                        
                        except Exception as e:
                            st.warning(f"⚠️ Unable to preview filter results: {str(e)}")
                
                # Results section
                if 'filtered_data' in st.session_state:
                    st.subheader("📈 Filtered Results")
                    
                    filtered_df = st.session_state.filtered_data
                    
                    # Results statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Records", len(df_to_modify))
                    with col2:
                        st.metric("Filtered Records", len(filtered_df))
                    with col3:
                        reduction_pct = ((len(df_to_modify) - len(filtered_df)) / len(df_to_modify) * 100) if len(df_to_modify) > 0 else 0
                        st.metric("Reduction %", f"{reduction_pct:.1f}%")
                    
                    # Show sample of filtered data
                    with st.expander("🔍 Preview Filtered Data (First 10 rows)"):
                        st.dataframe(filtered_df.head(10), use_container_width=True)
                    
                    # Comparison statistics
                    st.markdown("<div style='color: white;'><strong>📊 Before vs After Comparison:</strong></div>", unsafe_allow_html=True)
                    
                    comparison_data = []
                    for col in selected_columns:
                        if col in filtered_df.columns:
                            original_unique = df_to_modify[col].nunique()
                            filtered_unique = filtered_df[col].nunique()
                            
                            comparison_data.append({
                                'Column': col,
                                'Original Unique Values': original_unique,
                                'Filtered Unique Values': filtered_unique,
                                'Values Removed': original_unique - filtered_unique
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                    
                    # Export section
                    st.subheader("📥 Export Filtered Dataset")
                    
                    # Generate filename with timestamp
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"filtered_data_{timestamp}.csv"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download Filtered CSV",
                            filtered_df.to_csv(index=False).encode('utf-8'),
                            filename,
                            "text/csv",
                            help=f"Download filtered dataset with {len(filtered_df)} records"
                        )
                    
                    with col2:
                        # Option to clear results
                        if st.button("🗑️ Clear Results", key="clear_results"):
                            if 'filtered_data' in st.session_state:
                                del st.session_state.filtered_data
                            st.experimental_rerun()
    
    else:
        st.info("📁 Upload a CSV file or load data in the CSV Reviewer tab to start filtering")
        st.markdown("""
        **Dataset Modification Features:**
        - 🎯 **Filter Mode**: Choose to filter out or filter in specific values
        - 📋 **Multi-column filtering**: Apply filters across multiple categorical columns
        - 🔢 **Value counts**: See how many records each value has
        - 🔍 **Filter preview**: See the impact before applying changes
        - 📊 **Before/after comparison**: Understand the filtering impact
        - 📥 **Export functionality**: Download filtered dataset with timestamp
        
                 **Supported Logic:**
         - **AND between columns**: All column conditions must be met
         - **OR within columns**: Any selected value within a column matches
         - **Example**: `(len_group: b1 OR b2) AND (source: news OR wiki)`
         """)

### LLM Parse Data Analysis ###
with tab_llm_parse:
    st.title("🤖 LLM Parse Data Analysis")
    st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)
    
    st.markdown("Upload and analyze LLM training datasets generated by the parsing script.")
    
    # File upload
    st.subheader("📁 Upload Dataset")
    uploaded_arrow = st.file_uploader("Upload .arrow dataset file", type=["arrow"], key="llm_parse_arrow")
    
    if uploaded_arrow:
        try:
            # Import datasets library
            from datasets import Dataset
            import tempfile
            import os
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.arrow') as tmp_file:
                tmp_file.write(uploaded_arrow.read())
                tmp_path = tmp_file.name
            
            # Load dataset
            try:
                dataset = Dataset.from_file(tmp_path)
                st.success(f"✅ Loaded dataset with {len(dataset)} samples")
                
                # Basic Statistics
                st.subheader("📊 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(dataset))
                with col2:
                    st.metric("Columns", len(dataset.column_names))
                with col3:
                    st.metric("Features", ", ".join(dataset.column_names))
                
                # Prompt Type Distribution Analysis
                st.subheader("📈 Prompt Type Distribution")
                
                # Detect prompt type from sample content
                prompt_type_counts = {}
                format_patterns = {
                    'alpaca': '### Instruction:',
                    'gemma': '<start_of_turn>',
                    'chatml': '<|im_start|>'
                }
                
                # Count full dataset by format
                for i in range(len(dataset)):
                    sample = dataset[i]
                    
                    # Check if text field exists and is not null
                    if 'text' in sample and sample['text'] is not None:
                        text_content = str(sample['text'])
                        if text_content.strip():
                            format_detected = False
                            for fmt, pattern in format_patterns.items():
                                if pattern in text_content:
                                    prompt_type_counts[fmt] = prompt_type_counts.get(fmt, 0) + 1
                                    format_detected = True
                                    break
                            # If text exists but no pattern matched, count as unknown
                            if not format_detected:
                                prompt_type_counts['unknown'] = prompt_type_counts.get('unknown', 0) + 1
                        else:
                            # Empty text
                            prompt_type_counts['null_samples'] = prompt_type_counts.get('null_samples', 0) + 1
                    else:
                        # NULL text field - count as null
                        prompt_type_counts['null_samples'] = prompt_type_counts.get('null_samples', 0) + 1
                
                if prompt_type_counts:
                    # Display metrics
                    num_cols = min(len(prompt_type_counts), 4)  # Max 4 columns
                    cols = st.columns(num_cols)
                    for idx, (fmt, count) in enumerate(prompt_type_counts.items()):
                        with cols[idx % num_cols]:
                            pct = (count / len(dataset)) * 100
                            label = fmt.replace('_', ' ').title()
                            st.metric(f"{label}", f"{count} ({pct:.1f}%)")
                    
                    # Bar chart
                    prompt_df = pd.DataFrame(list(prompt_type_counts.items()), columns=['Prompt Type', 'Count'])
                    prompt_df = prompt_df.sort_values('Count', ascending=False)
                    st.bar_chart(prompt_df.set_index('Prompt Type')['Count'])
                    
                    # Show warning if null samples exist
                    if 'null_samples' in prompt_type_counts:
                        st.warning(f"⚠️ Found {prompt_type_counts['null_samples']} samples with NULL/empty text and no instruction field")
                else:
                    st.info("ℹ️ Unable to detect prompt format distribution")
                
                # Sample Browser
                st.subheader("🔍 Sample Browser")
                
                sample_idx = st.number_input("Sample Index", 0, len(dataset)-1, 0, 1, key="llm_sample_idx")
                
                if sample_idx < len(dataset):
                    sample = dataset[sample_idx]
                    
                    # Detect sample format
                    sample_format = "Unknown"
                    if 'text' in sample and sample['text'] is not None:
                        text_str = str(sample['text'])
                        if text_str.strip():
                            # Check for format patterns
                            if '### Instruction:' in text_str:
                                sample_format = "Alpaca"
                            elif '<start_of_turn>' in text_str:
                                sample_format = "Gemma"
                            elif '<|im_start|>' in text_str:
                                sample_format = "ChatML"
                            else:
                                sample_format = "Unknown Format"
                        else:
                            sample_format = "Empty"
                    elif 'text' in sample and sample['text'] is None:
                        sample_format = "NULL"
                    
                    st.markdown(f"<div style='color: white;'><strong>Sample {sample_idx + 1}/{len(dataset)} - Format: {sample_format}</strong></div>", unsafe_allow_html=True)
                    
                    # Display based on format
                    if sample_format == "NULL" or sample_format == "Empty":
                        st.error(f"⚠️ This sample has NULL or empty text content!")
                        st.markdown("<div style='color: #ff6b6b;'><strong>Issue:</strong> Sample is missing valid content. This may cause training errors.</div>", unsafe_allow_html=True)
                    
                    elif sample_format == "Alpaca" and 'instruction' in sample and sample['instruction']:
                        # Alpaca format with separate fields - show both formatted text and fields
                        st.markdown("<div style='color: white;'><strong>📝 Alpaca Format (Structured View):</strong></div>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("<div style='color: #ffd700;'><strong>Instruction:</strong></div>", unsafe_allow_html=True)
                            st.text_area("", sample.get('instruction', ''), height=100, key=f"inst_{sample_idx}")
                        
                        with col2:
                            st.markdown("<div style='color: #ffd700;'><strong>Input:</strong></div>", unsafe_allow_html=True)
                            st.text_area("", sample.get('input', ''), height=100, key=f"inp_{sample_idx}")
                        
                        st.markdown("<div style='color: #ffd700;'><strong>Output:</strong></div>", unsafe_allow_html=True)
                        st.text_area("", sample.get('output', ''), height=150, key=f"out_{sample_idx}")
                        
                        # Also show formatted text
                        with st.expander("📄 View as Formatted Text"):
                            st.text_area("Formatted Text", str(sample['text']), height=400, key=f"formatted_alpaca_{sample_idx}")
                    
                    elif 'text' in sample and sample['text'] is not None:
                        # Formatted sample (Gemma/ChatML/Alpaca without fields)
                        st.markdown(f"<div style='color: white;'><strong>📝 {sample_format} Formatted Text:</strong></div>", unsafe_allow_html=True)
                        st.text_area("Sample Text", str(sample['text']), height=400, key=f"formatted_{sample_idx}")
                    
                    # Raw JSON view
                    with st.expander("🔧 Raw Sample Data"):
                        st.json(sample)
                
                # Text Length Analysis
                st.subheader("📏 Text Length Analysis")
                
                text_lengths = []
                null_count = 0
                
                for i in range(len(dataset)):
                    sample = dataset[i]
                    if 'text' in sample and sample['text'] is not None:
                        text_str = str(sample['text'])
                        if text_str.strip():
                            text_lengths.append(len(text_str.split()))
                        else:
                            null_count += 1
                    elif 'instruction' in sample and sample['instruction']:
                        # Combine all fields for length
                        combined = f"{sample.get('instruction', '')} {sample.get('input', '')} {sample.get('output', '')}"
                        text_lengths.append(len(combined.split()))
                    else:
                        null_count += 1
                
                if text_lengths:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Length", f"{sum(text_lengths)/len(text_lengths):.1f} tokens")
                    with col2:
                        st.metric("Min Length", f"{min(text_lengths)} tokens")
                    with col3:
                        st.metric("Max Length", f"{max(text_lengths)} tokens")
                    with col4:
                        st.metric("Valid Samples", f"{len(text_lengths)}/{len(dataset)}")
                    
                    if null_count > 0:
                        st.warning(f"⚠️ Excluded {null_count} NULL/empty samples from length analysis")
                    
                    # Length distribution histogram
                    st.markdown("<div style='color: white;'><strong>Length Distribution:</strong></div>", unsafe_allow_html=True)
                    length_series = pd.Series(text_lengths)
                    length_hist = length_series.value_counts().sort_index()
                    st.bar_chart(length_hist)
                else:
                    st.error("❌ No valid samples found for length analysis")
                
                # Export sample preview
                st.subheader("📥 Export")
                
                # Export first 10 samples as JSON
                sample_preview = []
                for i in range(min(10, len(dataset))):
                    sample_preview.append(dataset[i])
                
                preview_json = json.dumps(sample_preview, ensure_ascii=False, indent=2).encode('utf-8')
                st.download_button(
                    "📥 Download Sample Preview (First 10)",
                    preview_json,
                    f"llm_dataset_preview.json",
                    "application/json",
                    help="Download first 10 samples as JSON"
                )
                
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
                # Note: Temp file cleanup handled by OS on session end
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("📁 Upload a .arrow dataset file to start analysis")
        st.markdown("""
        **Expected Format:**
        - `.arrow` files generated by the LLM parsing script
        - Contains either:
          - **Formatted samples**: `text` field with complete prompt
          - **Alpaca format**: `instruction`, `input`, `output` fields
        
        **Analysis Features:**
        - 📊 Dataset overview (total samples, columns)
        - 📈 Prompt type distribution (Alpaca, Gemma, ChatML)
        - 🔍 Interactive sample browser
        - 📏 Text length statistics and distribution
        - 📥 Export sample previews
        
        **Use Case:**
        Verify parsed LLM training data before fine-tuning to ensure correct formatting and distribution.
        """)


### Corrupted Level ###
with tab_corruption_level:
    st.title("📊 Corrupted Level Analysis & Control")
    st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)
    
    st.markdown("""
    **Analyze and control corruption levels:**
    - View corruption statistics (absolute and relative)
    - Reduce corruption level by replacing corrupted claims with originals
    - Choose selection strategy (random or max transformations)
    """)
    
    uploaded_file_corruption = st.file_uploader("Upload JSON file", type=["json"], key="corruption_uploader")
    if uploaded_file_corruption:
        data_corruption = json.load(uploaded_file_corruption)
        
        # Corruption Reduction Control Panel
        st.subheader("🎚️ Corruption Level Reduction")
        
        with st.expander("⚙️ Reduce Corruption Level (Modify Dataset)", expanded=False):
            st.markdown("<div style='color: white;'><strong>Select reduction mode:</strong></div>", unsafe_allow_html=True)
            
            # Choose reduction mode
            reduction_mode = st.radio(
                "Reduction Mode",
                options=['single', 'plan'],
                format_func=lambda x: "Single Reduction (one target % for all summaries)" if x == 'single' else "Plan Reduction (multiple parts with different %)",
                key="reduction_mode"
            )
            
            st.markdown("---")
            
            if reduction_mode == 'single':
                # Original single reduction UI
                st.markdown("<div style='color: white;'><strong>Configure single reduction:</strong></div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    target_corruption = st.slider(
                        "Target Corruption %",
                        min_value=0,
                        max_value=100,
                        value=50,
                        step=5,
                        help="Target percentage of claims to corrupt in each summary"
                    )
                
                with col2:
                    strategy = st.selectbox(
                        "Claim Selection Strategy",
                        options=['random', 'max_transformation', 'hebrew_focus'],
                        format_func=lambda x: {
                            'random': 'Random',
                            'max_transformation': 'Max Transformations (most error types per claim)',
                            'hebrew_focus': 'Hebrew Focus (prioritize Hebrew linguistic errors)'
                        }[x],
                        help="Strategy for selecting which corrupted claims to keep after reduction"
                    )
                
                total_summaries = sum(len(doc.get('correctness_corrupted_summaries', [])) for doc in data_corruption)
                num_summaries_to_process = st.number_input(
                    "Number of summaries to process",
                    min_value=1,
                    max_value=total_summaries,
                    value=min(10, total_summaries),
                    help=f"Process first N summaries (total available: {total_summaries})"
                )
                
                # Apply button for single reduction
                if st.button("🔧 Apply Single Reduction", type="primary", key="apply_single"):
                    modified_data = []
                    processed_count = 0
                    
                    with st.spinner(f"Processing {num_summaries_to_process} summaries..."):
                        for doc in data_corruption:
                            modified_doc = doc.copy()
                            modified_summaries = []
                            
                            for corrupted_summary in doc.get('correctness_corrupted_summaries', []):
                                if processed_count >= num_summaries_to_process:
                                    modified_summaries.append(corrupted_summary)
                                else:
                                    modified_summary = reduce_corruption_level(
                                        corrupted_summary,
                                        target_corruption,
                                        strategy
                                    )
                                    modified_summaries.append(modified_summary)
                                    processed_count += 1
                            
                            modified_doc['correctness_corrupted_summaries'] = modified_summaries
                            modified_data.append(modified_doc)
                    
                    st.success(f"✅ Successfully processed {processed_count} summaries!")
                    
                    # Store in session state for display
                    st.session_state['modified_data'] = modified_data
                    st.session_state['reduction_applied'] = True
                    st.session_state['processed_count'] = processed_count
                    st.session_state['applied_reduction_mode'] = 'single'
                    st.session_state['config_params'] = {
                        'target_corruption': target_corruption,
                        'strategy': strategy,
                        'processed_count': processed_count,
                        'dataset_name': uploaded_file_corruption.name if uploaded_file_corruption else 'dataset'
                    }
            
            else:  # plan reduction
                st.markdown("<div style='color: white;'><strong>Configure plan reduction:</strong></div>", unsafe_allow_html=True)
                
                total_summaries = sum(len(doc.get('correctness_corrupted_summaries', [])) for doc in data_corruption)
                st.info(f"📊 Total summaries available: {total_summaries}")
                
                # Number of parts
                num_parts = st.number_input(
                    "Number of parts",
                    min_value=2,
                    max_value=10,
                    value=3,
                    help="How many different corruption levels to create"
                )
                
                # Configure each part
                parts_config = []
                total_configured = 0
                
                for i in range(num_parts):
                    st.markdown(f"<div style='color: #ffd700;'><strong>Part {i+1}:</strong></div>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        part_samples = st.number_input(
                            f"Samples (Part {i+1})",
                            min_value=0,
                            max_value=total_summaries,
                            value=10 if i == 0 else (30 if i == 1 else max(0, total_summaries - total_configured)),
                            key=f"part_samples_{i}"
                        )
                    
                    with col2:
                        part_corruption = st.slider(
                            f"Target % (Part {i+1})",
                            min_value=0,
                            max_value=100,
                            value=25 if i == 0 else (70 if i == 1 else 90),
                            step=5,
                            key=f"part_corruption_{i}"
                        )
                    
                    with col3:
                        part_strategy = st.selectbox(
                            f"Strategy (Part {i+1})",
                            options=['random', 'max_transformation', 'hebrew_focus'],
                            format_func=lambda x: {
                                'random': 'Random',
                                'max_transformation': 'Max Transform',
                                'hebrew_focus': 'Hebrew Focus'
                            }[x],
                            key=f"part_strategy_{i}"
                        )
                    
                    parts_config.append({
                        'samples': part_samples,
                        'corruption': part_corruption,
                        'strategy': part_strategy
                    })
                    total_configured += part_samples
                
                # Validation
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Configured", total_configured)
                with col2:
                    st.metric("Available Summaries", total_summaries)
                
                if total_configured != total_summaries:
                    st.error(f"❌ Total configured samples ({total_configured}) must equal available summaries ({total_summaries})")
                else:
                    st.success(f"✅ Configuration valid!")
                
                # Apply button for plan reduction
                if st.button("🔧 Apply Plan Reduction", type="primary", key="apply_plan", disabled=(total_configured != total_summaries)):
                    import random
                    
                    with st.spinner(f"Applying plan reduction to {total_summaries} summaries..."):
                        # Flatten all summaries with their doc/summary indices
                        all_summaries = []
                        for doc_idx, doc in enumerate(data_corruption):
                            for summary_idx, corrupted_summary in enumerate(doc.get('correctness_corrupted_summaries', [])):
                                all_summaries.append({
                                    'doc_idx': doc_idx,
                                    'summary_idx': summary_idx,
                                    'summary': corrupted_summary
                                })
                        
                        # Randomly shuffle summaries
                        random.shuffle(all_summaries)
                        
                        # Assign summaries to parts
                        part_assignments = []
                        start_idx = 0
                        for part in parts_config:
                            end_idx = start_idx + part['samples']
                            part_summaries = all_summaries[start_idx:end_idx]
                            part_assignments.append({
                                'summaries': part_summaries,
                                'corruption': part['corruption'],
                                'strategy': part['strategy']
                            })
                            start_idx = end_idx
                        
                        # Apply reduction for each part
                        modified_data = [doc.copy() for doc in data_corruption]
                        for doc in modified_data:
                            doc['correctness_corrupted_summaries'] = doc.get('correctness_corrupted_summaries', [])[:]
                        
                        for part_idx, part in enumerate(part_assignments):
                            for item in part['summaries']:
                                doc_idx = item['doc_idx']
                                summary_idx = item['summary_idx']
                                original_summary = item['summary']
                                
                                # Apply reduction
                                modified_summary = reduce_corruption_level(
                                    original_summary,
                                    part['corruption'],
                                    part['strategy']
                                )
                                
                                # Update in modified_data
                                modified_data[doc_idx]['correctness_corrupted_summaries'][summary_idx] = modified_summary
                    
                    st.success(f"✅ Successfully applied plan reduction to {total_summaries} summaries!")
                    
                    # Store in session state for display
                    st.session_state['modified_data'] = modified_data
                    st.session_state['reduction_applied'] = True
                    st.session_state['processed_count'] = total_summaries
                    st.session_state['applied_reduction_mode'] = 'plan'
                    st.session_state['config_params'] = {
                        'parts_config': parts_config,
                        'total_summaries': total_summaries,
                        'dataset_name': uploaded_file_corruption.name if uploaded_file_corruption else 'dataset'
                    }
            
            # Show results if reduction was applied
            if st.session_state.get('reduction_applied', False) and 'modified_data' in st.session_state:
                modified_data = st.session_state['modified_data']
                processed_count = st.session_state.get('processed_count', 0)
                reduction_mode_stored = st.session_state.get('applied_reduction_mode', 'single')
                config_params = st.session_state.get('config_params', {})
                
                # Create ZIP file with dataset and metadata
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add modified dataset JSON
                    modified_json = json.dumps(modified_data, ensure_ascii=False, indent=2)
                    if reduction_mode_stored == 'single':
                        target_corruption = config_params.get('target_corruption', 'unknown')
                        json_filename = f"reduced_corruption_{target_corruption}pct.json"
                    else:
                        json_filename = "plan_reduced_corruption.json"
                    zip_file.writestr(json_filename, modified_json)
                    
                    # Generate and add metadata text file
                    metadata_text = generate_metadata_text(modified_data, reduction_mode_stored, config_params)
                    zip_file.writestr("metadata.txt", metadata_text)
                
                # Prepare ZIP for download
                zip_buffer.seek(0)
                zip_filename = f"corruption_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                
                # Show ZIP contents info
                st.info(f"📦 ZIP Package Contents:\n- {json_filename} (Modified dataset)\n- metadata.txt (Corruption statistics and configuration)")
                
                st.download_button(
                    "📥 Download Modified Dataset (ZIP)",
                    zip_buffer.getvalue(),
                    zip_filename,
                    "application/zip",
                    help="Download ZIP file containing modified dataset and metadata information"
                )
                
                # Show detailed statistics for modified data
                st.markdown("---")
                st.markdown("<div style='color: white;'><strong>📊 Modified Dataset Statistics:</strong></div>", unsafe_allow_html=True)
                
                # Calculate detailed statistics for modified data
                modified_corruption_details = []
                modified_corruption_levels = []
                modified_corruption_percentages = []
                
                for doc_idx, doc in enumerate(modified_data):
                    for summary_idx, corrupted_summary in enumerate(doc.get('correctness_corrupted_summaries', [])):
                        transformation_list = corrupted_summary.get('transformation_list', [])
                        actual_count = len(transformation_list)
                        stored_count = corrupted_summary.get('transformations_performed', 0)
                        corrupted_text = corrupted_summary.get('corrupted_summary', '')
                        total_claims = count_claims_sentences(corrupted_text)
                        corruption_percentage = (actual_count / total_claims * 100) if total_claims > 0 else 0.0
                        
                        modified_corruption_levels.append(actual_count)
                        modified_corruption_percentages.append(corruption_percentage)
                        modified_corruption_details.append({
                            'doc_idx': doc_idx,
                            'summary_idx': summary_idx,
                            'actual_transformations': actual_count,
                            'stored_transformations': stored_count,
                            'total_claims': total_claims,
                            'corruption_percentage': f"{corruption_percentage:.1f}%",
                            'inconsistent': '⚠️' if actual_count != stored_count else '✓'
                        })
                
                # Show summary metrics
                if modified_corruption_levels:
                    avg_corruption_mod = sum(modified_corruption_levels) / len(modified_corruption_levels)
                    min_corruption_mod = min(modified_corruption_levels)
                    max_corruption_mod = max(modified_corruption_levels)
                    avg_corruption_pct_mod = sum(modified_corruption_percentages) / len(modified_corruption_percentages)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Average Corruptions", f"{avg_corruption_mod:.2f}")
                    with col2:
                        st.metric("Min Corruptions", min_corruption_mod)
                    with col3:
                        st.metric("Max Corruptions", max_corruption_mod)
                    with col4:
                        st.metric("Total Summaries", len(modified_corruption_levels))
                    with col5:
                        st.metric("Avg Corruption %", f"{avg_corruption_pct_mod:.1f}%")
                    
                    # Show detailed table
                    with st.expander("📋 View Detailed Corruption Stats per Summary (Modified Data)", expanded=True):
                        modified_details_df = pd.DataFrame(modified_corruption_details)
                        modified_details_df = modified_details_df.sort_values('actual_transformations', ascending=False)
                        st.dataframe(modified_details_df, use_container_width=True)
                        
                        # Sample selector for reviewing modified samples
                        st.markdown("---")
                        st.markdown("<div style='color: white;'><strong>🔍 Review Sample Details:</strong></div>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_doc_idx_mod = st.selectbox(
                                "Select Document Index:",
                                options=sorted(modified_details_df['doc_idx'].unique()),
                                key="modified_doc_selector"
                            )
                        with col2:
                            # Filter summaries for selected document
                            available_summaries_mod = modified_details_df[modified_details_df['doc_idx'] == selected_doc_idx_mod]['summary_idx'].tolist()
                            selected_summary_idx_mod = st.selectbox(
                                "Select Summary Index:",
                                options=available_summaries_mod,
                                key="modified_summary_selector"
                            )
                        
                        # Display selected sample details
                        if selected_doc_idx_mod is not None and selected_summary_idx_mod is not None:
                            selected_doc_mod = modified_data[selected_doc_idx_mod]
                            selected_summary_mod = selected_doc_mod.get('correctness_corrupted_summaries', [])[selected_summary_idx_mod]
                            
                            # Manual count (GROUND TRUTH)
                            actual_count_mod = len(selected_summary_mod.get('transformation_list', []))
                            stored_count_mod = selected_summary_mod.get('transformations_performed', 0)
                            
                            # Count total claims using SENTENCES method
                            corrupted_text_mod = selected_summary_mod.get('corrupted_summary', '')
                            total_claims_mod = count_claims_sentences(corrupted_text_mod)
                            corruption_pct_mod = (actual_count_mod / total_claims_mod * 100) if total_claims_mod > 0 else 0.0
                            
                            # Show metrics
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Actual Claims Corrupted", f"{actual_count_mod}")
                            with col2:
                                st.metric("Stored Value", f"{stored_count_mod}")
                            with col3:
                                st.metric("Total Claims", f"{total_claims_mod}")
                            with col4:
                                st.metric("Corruption %", f"{corruption_pct_mod:.1f}%")
                            with col5:
                                match_status_mod = "✅ Match" if actual_count_mod == stored_count_mod else "⚠️ Mismatch"
                                st.metric("Status", match_status_mod)
                            
                            # Show source article
                            st.markdown("<div style='color: white;'><strong>📰 Source Article:</strong></div>", unsafe_allow_html=True)
                            st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 15px; border-radius: 5px; max-height: 300px; overflow-y: auto;'>{selected_doc_mod.get('source_article', '')}</div>", unsafe_allow_html=True)
                            
                            # Show gold summary
                            st.markdown("<div style='color: white;'><strong>🏆 Gold Summary:</strong></div>", unsafe_allow_html=True)
                            st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #4CAF50;'>{selected_doc_mod.get('gold_summary', '')}</div>", unsafe_allow_html=True)
                            
                            # Show modified corrupted summary
                            st.markdown("<div style='color: white;'><strong>🔴 Modified Corrupted Summary (After Reduction):</strong></div>", unsafe_allow_html=True)
                            st.markdown(f"<div dir='rtl' style='background-color: #4d1a1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #f44336;'>{selected_summary_mod.get('corrupted_summary', '')}</div>", unsafe_allow_html=True)
                            
                            # Show diff
                            st.markdown("<div style='color: white;'><strong>⚡ Difference Visualization:</strong></div>", unsafe_allow_html=True)
                            diff_html_mod = highlight_diff(selected_doc_mod.get('gold_summary', ''), selected_summary_mod.get('corrupted_summary', ''))
                            st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 15px; border-radius: 5px; border: 1px solid #666;'>{diff_html_mod}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Corruption Level Statistics
        st.subheader("📊 Corruption Level Statistics")
        
        # Calculate corruption statistics for all summaries
        all_corruption_levels = []
        all_corruption_percentages = []
        corruption_details = []
        inconsistencies_found = 0
        
        for doc_idx, doc in enumerate(data_corruption):
            for summary_idx, corrupted_summary in enumerate(doc.get('correctness_corrupted_summaries', [])):
                # Manual count from transformation_list (GROUND TRUTH)
                transformation_list = corrupted_summary.get('transformation_list', [])
                actual_count = len(transformation_list)
                
                # Stored value (may be inconsistent)
                stored_count = corrupted_summary.get('transformations_performed', 0)
                
                # Count total claims in corrupted summary using SENTENCES method
                corrupted_text = corrupted_summary.get('corrupted_summary', '')
                total_claims = count_claims_sentences(corrupted_text)
                
                # Calculate corruption percentage
                corruption_percentage = (actual_count / total_claims * 100) if total_claims > 0 else 0.0
                
                # Check for inconsistency
                is_inconsistent = (actual_count != stored_count)
                if is_inconsistent:
                    inconsistencies_found += 1
                
                all_corruption_levels.append(actual_count)
                all_corruption_percentages.append(corruption_percentage)
                corruption_details.append({
                    'doc_idx': doc_idx,
                    'summary_idx': summary_idx,
                    'actual_transformations': actual_count,
                    'stored_transformations': stored_count,
                    'total_claims': total_claims,
                    'corruption_percentage': f"{corruption_percentage:.1f}%",
                    'inconsistent': '⚠️' if is_inconsistent else '✓'
                })
        
        if all_corruption_levels:
            avg_corruption = sum(all_corruption_levels) / len(all_corruption_levels)
            min_corruption = min(all_corruption_levels)
            max_corruption = max(all_corruption_levels)
            avg_corruption_pct = sum(all_corruption_percentages) / len(all_corruption_percentages)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Average Corruptions", f"{avg_corruption:.2f}")
            with col2:
                st.metric("Min Corruptions", min_corruption)
            with col3:
                st.metric("Max Corruptions", max_corruption)
            with col4:
                st.metric("Total Summaries", len(all_corruption_levels))
            with col5:
                st.metric("Avg Corruption %", f"{avg_corruption_pct:.1f}%")
            with col6:
                inconsistency_label = "Inconsistencies ⚠️" if inconsistencies_found > 0 else "Data Quality ✓"
                st.metric(inconsistency_label, inconsistencies_found)
            
            # Show distribution chart
            st.markdown("<div style='color: white;'><strong>Corruption Level Distribution:</strong></div>", unsafe_allow_html=True)
            corruption_df_chart = pd.DataFrame({'Transformations Performed': all_corruption_levels})
            st.bar_chart(corruption_df_chart['Transformations Performed'].value_counts().sort_index())
            
            # Show percentage distribution chart
            st.markdown("<div style='color: white;'><strong>Corruption Percentage Distribution:</strong></div>", unsafe_allow_html=True)
            corruption_pct_chart = pd.DataFrame({'Corruption Percentage': all_corruption_percentages})
            st.bar_chart(corruption_pct_chart['Corruption Percentage'].value_counts().sort_index())
            
            # Show detailed table in expander
            with st.expander("📋 View Detailed Corruption Stats per Summary", expanded=True):
                details_df = pd.DataFrame(corruption_details)
                details_df = details_df.sort_values('actual_transformations', ascending=False)
                
                # Show warning if inconsistencies found
                if inconsistencies_found > 0:
                    st.warning(f"⚠️ Found {inconsistencies_found} inconsistencies between stored and actual counts. Using ACTUAL counts from transformation_list.")
                else:
                    st.success("✅ All stored counts match actual transformation_list lengths.")
                
                st.dataframe(details_df, use_container_width=True)
                
                # Sample selector for reviewing article and summary
                st.markdown("---")
                st.markdown("<div style='color: white;'><strong>🔍 Review Sample Details:</strong></div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_doc_idx = st.selectbox(
                        "Select Document Index:",
                        options=sorted(details_df['doc_idx'].unique()),
                        key="corruption_level_doc_selector"
                    )
                with col2:
                    # Filter summaries for selected document
                    available_summaries = details_df[details_df['doc_idx'] == selected_doc_idx]['summary_idx'].tolist()
                    selected_summary_idx = st.selectbox(
                        "Select Summary Index:",
                        options=available_summaries,
                        key="corruption_level_summary_selector"
                    )
                
                # Display selected sample details
                if selected_doc_idx is not None and selected_summary_idx is not None:
                    selected_doc = data_corruption[selected_doc_idx]
                    selected_summary = selected_doc.get('correctness_corrupted_summaries', [])[selected_summary_idx]
                    
                    # Manual count (GROUND TRUTH)
                    actual_count = len(selected_summary.get('transformation_list', []))
                    stored_count = selected_summary.get('transformations_performed', 0)
                    
                    # Count total claims using SENTENCES method
                    corrupted_text = selected_summary.get('corrupted_summary', '')
                    total_claims = count_claims_sentences(corrupted_text)
                    corruption_pct = (actual_count / total_claims * 100) if total_claims > 0 else 0.0
                    
                    # Show metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Actual Claims Corrupted", f"{actual_count}")
                    with col2:
                        st.metric("Stored Value", f"{stored_count}")
                    with col3:
                        st.metric("Total Claims", f"{total_claims}")
                    with col4:
                        st.metric("Corruption %", f"{corruption_pct:.1f}%")
                    with col5:
                        match_status = "✅ Match" if actual_count == stored_count else "⚠️ Mismatch"
                        st.metric("Status", match_status)
                    
                    # Show source article
                    st.markdown("<div style='color: white;'><strong>📰 Source Article:</strong></div>", unsafe_allow_html=True)
                    st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 15px; border-radius: 5px; max-height: 300px; overflow-y: auto;'>{selected_doc.get('source_article', '')}</div>", unsafe_allow_html=True)
                    
                    # Show gold summary
                    st.markdown("<div style='color: white;'><strong>🏆 Gold Summary:</strong></div>", unsafe_allow_html=True)
                    st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #4CAF50;'>{selected_doc.get('gold_summary', '')}</div>", unsafe_allow_html=True)
                    
                    # Show corrupted summary
                    st.markdown("<div style='color: white;'><strong>🔴 Corrupted Summary:</strong></div>", unsafe_allow_html=True)
                    st.markdown(f"<div dir='rtl' style='background-color: #4d1a1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #f44336;'>{selected_summary.get('corrupted_summary', '')}</div>", unsafe_allow_html=True)
                    
                    # Show diff
                    st.markdown("<div style='color: white;'><strong>⚡ Difference Visualization:</strong></div>", unsafe_allow_html=True)
                    diff_html = highlight_diff(selected_doc.get('gold_summary', ''), selected_summary.get('corrupted_summary', ''))
                    st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 15px; border-radius: 5px; border: 1px solid #666;'>{diff_html}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No corruption statistics available")
    else:
        st.info("📁 Upload a JSON augmentation file to analyze corruption levels")
        st.markdown("""
        **Expected Format:**
        - JSON augmentation output with `correctness_corrupted_summaries`
        - Each summary should have `transformation_list` and optionally `transformations_performed`
        
        **Analysis Features:**
        - 📊 Absolute corruption metrics (min, max, average)
        - 📈 Relative corruption percentage (% of claims corrupted)
        - ⚠️ Inconsistency detection between stored and actual counts
        - 🔍 Sample-by-sample review with source, gold, and corrupted summaries
        - 📉 Distribution charts for both absolute and percentage corruption
        """)