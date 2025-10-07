import sys
import os
import json
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Mafat Dataset Analysis", layout="wide", page_icon="📊")

st.title("📊 Mafat Dataset Analysis")
st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)

st.markdown("Analyze JSONL datasets from both original format and Ollama-enhanced summary generation outputs.")

# Helper functions for JSONL parsing and analysis
@st.cache_data(show_spinner=False)
def load_jsonl_files(uploaded_files):
    """Load and parse multiple JSONL files"""
    all_records = []
    file_info = {}
    
    for file in uploaded_files:
        records = []
        try:
            # Reset file pointer
            file.seek(0)
            content = file.read().decode('utf-8')
            
            # Parse line by line
            for line_num, line in enumerate(content.strip().split('\n')):
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        record['_source_file'] = file.name
                        record['_line_number'] = line_num + 1
                        records.append(record)
                    except json.JSONDecodeError as e:
                        st.warning(f"⚠️ Skipped invalid JSON on line {line_num + 1} in {file.name}: {str(e)[:100]}")
                        continue
            
            file_info[file.name] = {
                'total_records': len(records),
                'format_detected': detect_format(records[0] if records else {})
            }
            all_records.extend(records)
            
        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {str(e)}")
            continue
    
    return all_records, file_info

def detect_format(record):
    """Detect if record is original format or Ollama-enhanced"""
    if 'metadata' in record and 'generative_summaries' in record.get('metadata', {}):
        return 'ollama_enhanced'
    elif 'text_raw' in record and 'summary' in record:
        return 'original'
    else:
        return 'unknown'

def extract_metadata_fields(records):
    """Extract all available metadata fields across records"""
    all_fields = set()
    metadata_fields = set()
    
    for record in records:
        all_fields.update(record.keys())
        if 'metadata' in record and isinstance(record['metadata'], dict):
            metadata_fields.update(record['metadata'].keys())
    
    return sorted(all_fields), sorted(metadata_fields)

def get_unique_values(records, field_path):
    """Get unique values for a field (supports nested paths like 'metadata.source')"""
    values = set()
    
    for record in records:
        if '.' in field_path:
            parts = field_path.split('.')
            current = record
            try:
                for part in parts:
                    current = current[part]
                if current is not None:
                    values.add(str(current))
            except (KeyError, TypeError):
                continue
        else:
            if field_path in record and record[field_path] is not None:
                values.add(str(record[field_path]))
    
    return sorted(list(values))

def apply_filters(records, filters):
    """Apply filters to records"""
    if not filters:
        return records
    
    filtered = []
    for record in records:
        match = True
        
        for field_path, selected_values in filters.items():
            if not selected_values:
                continue
                
            record_value = None
            if '.' in field_path:
                parts = field_path.split('.')
                current = record
                try:
                    for part in parts:
                        current = current[part]
                    record_value = str(current) if current is not None else None
                except (KeyError, TypeError):
                    record_value = None
            else:
                record_value = str(record[field_path]) if field_path in record and record[field_path] is not None else None
            
            if record_value not in selected_values:
                match = False
                break
        
        if match:
            filtered.append(record)
    
    return filtered

def search_in_text(records, search_terms, search_fields):
    """Search for terms in specified text fields"""
    if not search_terms or not search_fields:
        return records
    
    search_terms = [term.lower().strip() for term in search_terms.split() if term.strip()]
    if not search_terms:
        return records
    
    filtered = []
    for record in records:
        found = False
        
        for field in search_fields:
            text_content = ""
            if '.' in field:
                parts = field.split('.')
                current = record
                try:
                    for part in parts:
                        current = current[part]
                    text_content = str(current) if current is not None else ""
                except (KeyError, TypeError):
                    continue
            else:
                text_content = str(record.get(field, ""))
            
            text_content = text_content.lower()
            if any(term in text_content for term in search_terms):
                found = True
                break
        
        if found:
            filtered.append(record)
    
    return filtered

def calculate_word_counts(text):
    """Calculate word count for text"""
    return len(str(text).split()) if text else 0

def get_generative_summaries_info(record):
    """Extract generative summaries information"""
    summaries = []
    if 'metadata' in record and 'generative_summaries' in record['metadata']:
        for summary in record['metadata']['generative_summaries']:
            summaries.append({
                'model': summary.get('model', 'unknown'),
                'prompt': summary.get('prompt', 'unknown'),
                'content': summary.get('generative_summary', ''),
                'word_count': calculate_word_counts(summary.get('generative_summary', ''))
            })
    return summaries

# File upload
uploaded_files = st.file_uploader(
    "Upload JSONL file(s)", 
    type=["jsonl"], 
    accept_multiple_files=True, 
    key="mafat_upload"
)

if uploaded_files:
    # Load data
    with st.spinner("Loading and parsing JSONL files..."):
        records, file_info = load_jsonl_files(uploaded_files)
    
    if not records:
        st.error("❌ No valid records found in uploaded files")
        st.stop()
    
    st.success(f"✅ Loaded {len(records)} records from {len(uploaded_files)} file(s)")
    
    # File info summary
    with st.expander("📁 File Information"):
        for filename, info in file_info.items():
            st.markdown(f"**{filename}**: {info['total_records']} records, format: {info['format_detected']}")
    
    # Extract available fields
    all_fields, metadata_fields = extract_metadata_fields(records)
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Basic filters
    filters = {}
    
    # Source filter (metadata.source)
    if 'source' in metadata_fields:
        source_values = get_unique_values(records, 'metadata.source')
        if source_values:
            selected_sources = st.sidebar.multiselect(
                "📰 Source", 
                options=source_values,
                key="source_filter"
            )
            if selected_sources:
                filters['metadata.source'] = selected_sources
    
    # Annotator filter (metadata.annotator)
    if 'annotator' in metadata_fields:
        annotator_values = get_unique_values(records, 'metadata.annotator')
        if annotator_values:
            selected_annotators = st.sidebar.multiselect(
                "👤 Annotator", 
                options=annotator_values,
                key="annotator_filter"
            )
            if selected_annotators:
                filters['metadata.annotator'] = selected_annotators
    
    # Type filter (metadata.type)
    if 'type' in metadata_fields:
        type_values = get_unique_values(records, 'metadata.type')
        if type_values:
            selected_types = st.sidebar.multiselect(
                "🏷️ Type", 
                options=type_values,
                key="type_filter"
            )
            if selected_types:
                filters['metadata.type'] = selected_types
    
    # User filter
    if 'user' in all_fields:
        user_values = get_unique_values(records, 'user')
        if user_values:
            selected_users = st.sidebar.multiselect(
                "👥 User", 
                options=user_values,
                key="user_filter"
            )
            if selected_users:
                filters['user'] = selected_users
    
    # Model filter (for Ollama-enhanced records)
    model_values = set()
    for record in records:
        summaries = get_generative_summaries_info(record)
        for summary in summaries:
            model_values.add(summary['model'])
    
    if model_values:
        model_values = sorted(list(model_values))
        selected_models = st.sidebar.multiselect(
            "🤖 Model", 
            options=model_values,
            key="model_filter"
        )
    else:
        selected_models = []
    
    # Text search
    st.sidebar.subheader("🔍 Text Search")
    search_terms = st.sidebar.text_input(
        "Search in text fields:",
        placeholder="Enter search terms...",
        key="text_search"
    )
    
    search_fields = st.sidebar.multiselect(
        "Search in fields:",
        options=['text_raw', 'summary', 'metadata.ai_summary'],
        default=['text_raw', 'summary'],
        key="search_fields"
    )
    
    # Word count filters
    st.sidebar.subheader("📏 Word Count Filters")
    enable_word_filter = st.sidebar.checkbox("Enable word count filtering")
    
    if enable_word_filter:
        min_words = st.sidebar.number_input("Min words", min_value=0, value=0, key="min_words")
        max_words = st.sidebar.number_input("Max words", min_value=0, value=1000, key="max_words")
    
    # Apply filters
    filtered_records = records.copy()
    
    # Apply categorical filters
    if filters:
        filtered_records = apply_filters(filtered_records, filters)
    
    # Apply model filter
    if selected_models:
        model_filtered = []
        for record in filtered_records:
            summaries = get_generative_summaries_info(record)
            if any(summary['model'] in selected_models for summary in summaries):
                model_filtered.append(record)
        filtered_records = model_filtered
    
    # Apply text search
    if search_terms and search_fields:
        filtered_records = search_in_text(filtered_records, search_terms, search_fields)
    
    # Apply word count filter
    if enable_word_filter:
        word_filtered = []
        for record in filtered_records:
            text_raw_words = calculate_word_counts(record.get('text_raw', ''))
            summary_words = calculate_word_counts(record.get('summary', ''))
            
            if min_words <= max(text_raw_words, summary_words) <= max_words:
                word_filtered.append(record)
        filtered_records = word_filtered
    
    # Clear filters button
    if st.sidebar.button("🗑️ Clear All Filters"):
        st.rerun()
    
    # Main content tabs
    tab_overview, tab_browse, tab_compare, tab_export = st.tabs([
        "📊 Overview", "🔍 Browse", "⚖️ Compare", "📥 Export"
    ])
    
    with tab_overview:
        st.subheader("📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(records))
        with col2:
            st.metric("Filtered Records", len(filtered_records))
        with col3:
            st.metric("Files Loaded", len(uploaded_files))
        with col4:
            reduction_pct = ((len(records) - len(filtered_records)) / len(records) * 100) if len(records) > 0 else 0
            st.metric("Filtered Out %", f"{reduction_pct:.1f}%")
        
        if filtered_records:
            # Format distribution
            format_counts = {}
            for record in filtered_records:
                fmt = detect_format(record)
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Format Distribution**")
                format_df = pd.DataFrame(list(format_counts.items()), columns=['Format', 'Count'])
                st.bar_chart(format_df.set_index('Format')['Count'])
            
            with col2:
                # Source distribution
                if 'metadata.source' in filters or not filters:
                    sources = {}
                    for record in filtered_records:
                        source = record.get('metadata', {}).get('source', 'unknown')
                        sources[source] = sources.get(source, 0) + 1
                    
                    if sources:
                        st.markdown("**📰 Source Distribution**")
                        source_df = pd.DataFrame(list(sources.items()), columns=['Source', 'Count'])
                        st.bar_chart(source_df.set_index('Source')['Count'])
            
            # Model usage (for Ollama-enhanced records)
            model_usage = {}
            for record in filtered_records:
                summaries = get_generative_summaries_info(record)
                for summary in summaries:
                    model = summary['model']
                    model_usage[model] = model_usage.get(model, 0) + 1
            
            if model_usage:
                st.markdown("**🤖 Model Usage Distribution**")
                model_df = pd.DataFrame(list(model_usage.items()), columns=['Model', 'Count'])
                st.bar_chart(model_df.set_index('Model')['Count'])
            
            # Text length analysis
            st.markdown("**📏 Text Length Analysis**")
            
            text_lengths = []
            summary_lengths = []
            
            for record in filtered_records:
                text_raw = record.get('text_raw', '')
                summary = record.get('summary', '')
                
                text_lengths.append(calculate_word_counts(text_raw))
                summary_lengths.append(calculate_word_counts(summary))
            
            col1, col2 = st.columns(2)
            with col1:
                if text_lengths:
                    avg_text = sum(text_lengths) / len(text_lengths)
                    st.metric("Avg Article Length", f"{avg_text:.1f} words")
                    st.metric("Article Range", f"{min(text_lengths)}-{max(text_lengths)} words")
            
            with col2:
                if summary_lengths:
                    avg_summary = sum(summary_lengths) / len(summary_lengths)
                    st.metric("Avg Summary Length", f"{avg_summary:.1f} words")
                    st.metric("Summary Range", f"{min(summary_lengths)}-{max(summary_lengths)} words")
    
    with tab_browse:
        st.subheader("🔍 Browse Records")
        
        if not filtered_records:
            st.warning("⚠️ No records match current filters")
        else:
            # Record navigation
            record_idx = st.number_input(
                "Record #", 
                min_value=1, 
                max_value=len(filtered_records), 
                value=1, 
                step=1
            ) - 1
            
            current_record = filtered_records[record_idx]
            
            # Navigation buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("⏮️ First") and record_idx > 0:
                    st.rerun()
            with col2:
                if st.button("⏪ Previous") and record_idx > 0:
                    st.rerun()
            with col3:
                if st.button("⏩ Next") and record_idx < len(filtered_records) - 1:
                    st.rerun()
            with col4:
                if st.button("⏭️ Last") and record_idx < len(filtered_records) - 1:
                    st.rerun()
            
            # Record details
            st.markdown(f"**Record {record_idx + 1} of {len(filtered_records)}**")
            
            # Basic info
            with st.expander("📄 Basic Information", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**File:** {current_record.get('_source_file', 'unknown')}")
                    st.markdown(f"**Line:** {current_record.get('_line_number', 'unknown')}")
                    st.markdown(f"**Format:** {detect_format(current_record)}")
                
                with col2:
                    if 'metadata' in current_record:
                        metadata = current_record['metadata']
                        st.markdown(f"**Source:** {metadata.get('source', 'N/A')}")
                        st.markdown(f"**Annotator:** {metadata.get('annotator', 'N/A')}")
                        st.markdown(f"**Type:** {metadata.get('type', 'N/A')}")
            
            # Text content
            with st.expander("📰 Article Text", expanded=True):
                text_raw = current_record.get('text_raw', '')
                word_count = calculate_word_counts(text_raw)
                st.markdown(f"**Word Count:** {word_count}")
                st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 15px; border-radius: 5px; max-height: 300px; overflow-y: auto;'>{text_raw}</div>", unsafe_allow_html=True)
            
            # Original summary
            with st.expander("📝 Original Summary", expanded=True):
                summary = current_record.get('summary', '')
                summary_words = calculate_word_counts(summary)
                st.markdown(f"**Word Count:** {summary_words}")
                st.markdown(f"<div dir='rtl' style='background-color: #1a4d1a; color: white; padding: 15px; border-radius: 5px; border: 2px solid #4CAF50;'>{summary}</div>", unsafe_allow_html=True)
            
            # AI Summary (if available)
            if 'metadata' in current_record and 'ai_summary' in current_record['metadata']:
                with st.expander("🤖 AI Summary"):
                    ai_summary = current_record['metadata']['ai_summary']
                    ai_words = calculate_word_counts(ai_summary)
                    st.markdown(f"**Word Count:** {ai_words}")
                    st.markdown(f"<div dir='rtl' style='background-color: #4d1a4d; color: white; padding: 15px; border-radius: 5px; border: 2px solid #9C27B0;'>{ai_summary}</div>", unsafe_allow_html=True)
            
            # Generative Summaries (Ollama-enhanced)
            summaries = get_generative_summaries_info(current_record)
            if summaries:
                with st.expander("🚀 Generative Summaries", expanded=True):
                    for i, summary in enumerate(summaries):
                        st.markdown(f"**Summary {i+1}**: {summary['model']} | {summary['prompt']} | {summary['word_count']} words")
                        st.markdown(f"<div dir='rtl' style='background-color: #4d4d1a; color: white; padding: 10px; border-radius: 5px; border: 1px solid #FFC107; margin: 5px 0;'>{summary['content']}</div>", unsafe_allow_html=True)
            
            # Raw JSON
            with st.expander("🔧 Raw JSON"):
                st.json(current_record)
    
    with tab_compare:
        st.subheader("⚖️ Summary Comparison")
        
        if not filtered_records:
            st.warning("⚠️ No records match current filters")
        else:
            # Select record for comparison
            record_idx = st.selectbox(
                "Select record to compare:", 
                range(len(filtered_records)),
                format_func=lambda x: f"Record {x+1} ({filtered_records[x].get('_source_file', 'unknown')})"
            )
            
            record = filtered_records[record_idx]
            
            # Get all available summaries
            summaries_to_compare = []
            
            # Original summary
            if 'summary' in record:
                summaries_to_compare.append({
                    'name': 'Original Summary',
                    'content': record['summary'],
                    'type': 'original',
                    'word_count': calculate_word_counts(record['summary'])
                })
            
            # AI summary
            if 'metadata' in record and 'ai_summary' in record['metadata']:
                summaries_to_compare.append({
                    'name': 'AI Summary',
                    'content': record['metadata']['ai_summary'],
                    'type': 'ai',
                    'word_count': calculate_word_counts(record['metadata']['ai_summary'])
                })
            
            # Generative summaries
            gen_summaries = get_generative_summaries_info(record)
            for i, gen_summary in enumerate(gen_summaries):
                summaries_to_compare.append({
                    'name': f'Generated ({gen_summary["model"]})',
                    'content': gen_summary['content'],
                    'type': 'generated',
                    'word_count': gen_summary['word_count'],
                    'model': gen_summary['model']
                })
            
            if len(summaries_to_compare) < 2:
                st.warning("⚠️ Need at least 2 summaries to compare")
            else:
                st.markdown(f"**Found {len(summaries_to_compare)} summaries for comparison**")
                
                # Display summaries side by side
                cols = st.columns(min(len(summaries_to_compare), 3))
                
                for i, summary in enumerate(summaries_to_compare[:3]):  # Show up to 3 at once
                    with cols[i]:
                        st.markdown(f"**{summary['name']}**")
                        st.markdown(f"*{summary['word_count']} words*")
                        
                        # Color coding by type
                        if summary['type'] == 'original':
                            bg_color = '#1a4d1a'
                            border_color = '#4CAF50'
                        elif summary['type'] == 'ai':
                            bg_color = '#4d1a4d'
                            border_color = '#9C27B0'
                        else:
                            bg_color = '#4d4d1a'
                            border_color = '#FFC107'
                        
                        st.markdown(f"<div dir='rtl' style='background-color: {bg_color}; color: white; padding: 10px; border-radius: 5px; border: 2px solid {border_color}; height: 200px; overflow-y: auto;'>{summary['content']}</div>", unsafe_allow_html=True)
                
                # If more than 3 summaries, show the rest
                if len(summaries_to_compare) > 3:
                    with st.expander("Additional Summaries"):
                        for summary in summaries_to_compare[3:]:
                            st.markdown(f"**{summary['name']}** ({summary['word_count']} words)")
                            st.markdown(f"<div dir='rtl' style='background-color: #2d2d2d; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;'>{summary['content']}</div>", unsafe_allow_html=True)
    
    with tab_export:
        st.subheader("📥 Export Options")
        
        if not filtered_records:
            st.warning("⚠️ No records to export")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Export Filtered Dataset**")
                st.markdown(f"Export {len(filtered_records)} filtered records")
                
                # Export as JSONL
                jsonl_data = '\n'.join(json.dumps(record, ensure_ascii=False) for record in filtered_records)
                st.download_button(
                    "📥 Download as JSONL",
                    jsonl_data.encode('utf-8'),
                    "filtered_dataset.jsonl",
                    "application/json",
                    help="Download filtered records in JSONL format"
                )
                
                # Export as JSON
                json_data = json.dumps(filtered_records, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 Download as JSON",
                    json_data.encode('utf-8'),
                    "filtered_dataset.json",
                    "application/json",
                    help="Download filtered records as JSON array"
                )
            
            with col2:
                st.markdown("**📋 Export Summary Statistics**")
                
                # Generate summary statistics
                stats = {
                    'total_records': len(records),
                    'filtered_records': len(filtered_records),
                    'filters_applied': bool(filters or selected_models or search_terms),
                    'file_info': file_info,
                    'format_distribution': {},
                    'source_distribution': {},
                    'model_usage': {}
                }
                
                # Calculate distributions
                for record in filtered_records:
                    fmt = detect_format(record)
                    stats['format_distribution'][fmt] = stats['format_distribution'].get(fmt, 0) + 1
                    
                    source = record.get('metadata', {}).get('source', 'unknown')
                    stats['source_distribution'][source] = stats['source_distribution'].get(source, 0) + 1
                    
                    summaries = get_generative_summaries_info(record)
                    for summary in summaries:
                        model = summary['model']
                        stats['model_usage'][model] = stats['model_usage'].get(model, 0) + 1
                
                stats_json = json.dumps(stats, ensure_ascii=False, indent=2)
                st.download_button(
                    "📊 Download Statistics",
                    stats_json.encode('utf-8'),
                    "dataset_statistics.json",
                    "application/json",
                    help="Download analysis statistics"
                )

else:
    st.info("📁 Upload JSONL file(s) to begin analysis")
    st.markdown("""
    **Supported Formats:**
    - **Original Format**: Records with `text_raw`, `summary`, `metadata` fields
    - **Ollama-Enhanced Format**: Original format + `metadata.generative_summaries` array
    
    **Features:**
    - 🎛️ **Advanced Filtering**: Filter by source, annotator, type, user, model, and text search
    - 🔍 **Record Browser**: Navigate through records with detailed views
    - ⚖️ **Summary Comparison**: Side-by-side comparison of different summary types
    - 📊 **Statistics Dashboard**: Distribution analysis and metrics
    - 📥 **Export Options**: Download filtered data and statistics
    - 📏 **Word Count Analysis**: Filter and analyze by text length
    """) 