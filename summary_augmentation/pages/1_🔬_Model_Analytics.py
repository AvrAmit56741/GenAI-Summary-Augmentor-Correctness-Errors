import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# Configure the page
st.set_page_config(page_title="Model Analytics", layout="wide", page_icon="🔬")

# Main page
st.title("🔬 Model Analytics")
st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)

st.markdown("Advanced analytics and performance evaluation for Hebrew NLI models.")

# Create internal tabs for Model Analytics
tab_multilabel, tab_binary = st.tabs([
    "🏷️ Multilabel Classifier", "🎯 Binary Classifier"
])

# ===== MULTILABEL CLASSIFIER TAB =====
with tab_multilabel:
    st.subheader("🏷️ Multilabel Classifier Analysis")
    st.markdown("Analyze multilabel classification performance for Hebrew NLI error types.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV with multilabel predictions", 
        type=["csv"], 
        key="multilabel_csv",
        help="CSV should contain metadata columns + prob_X, pred_X, label_X columns for each error type"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} records")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                # Count error type columns (prob_ columns)
                error_types = [col.replace('prob_', '') for col in df.columns if col.startswith('prob_')]
                st.metric("Error Types", len(error_types))
            
            # Validate required columns
            metadata_cols = ['index', 'premise', 'hypothesis', 'label', 'len_group', 'doc_id', 'source', 'error_types']
            missing_metadata = [col for col in metadata_cols if col not in df.columns]
            
            if missing_metadata:
                st.error(f"❌ Missing metadata columns: {missing_metadata}")
            elif not error_types:
                st.error("❌ No error type columns found. Expected columns starting with 'prob_'")
            else:
                # Validate that each error type has all three columns
                missing_triplets = []
                for error_type in error_types:
                    prob_col = f'prob_{error_type}'
                    pred_col = f'pred_{error_type}'
                    label_col = f'label_{error_type}'
                    
                    if prob_col not in df.columns:
                        missing_triplets.append(prob_col)
                    if pred_col not in df.columns:
                        missing_triplets.append(pred_col)
                    if label_col not in df.columns:
                        missing_triplets.append(label_col)
                
                if missing_triplets:
                    st.error(f"❌ Missing required columns: {missing_triplets}")
                else:
                    # Show sample data
                    with st.expander("🔍 Sample Data Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                    
                    # Perform analysis
                    st.subheader("📊 Performance Analysis")
                    
                    # Calculate metrics for each error type
                    error_type_metrics = []
                    all_predictions = []
                    all_labels = []
                    all_probabilities = []
                    
                    for error_type in error_types:
                        pred_col = f'pred_{error_type}'
                        label_col = f'label_{error_type}'
                        prob_col = f'prob_{error_type}'
                        
                        # Get predictions and labels
                        y_pred = df[pred_col].values
                        y_true = df[label_col].values
                        y_prob = df[prob_col].values
                        
                        # Calculate metrics
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        
                        # Calculate confidence analysis
                        correct_mask = (y_pred == y_true)
                        wrong_mask = (y_pred != y_true)
                        
                        avg_confidence_correct = y_prob[correct_mask].mean() if correct_mask.sum() > 0 else 0
                        avg_confidence_wrong = y_prob[wrong_mask].mean() if wrong_mask.sum() > 0 else 0
                        
                        error_type_metrics.append({
                            'Error Type': error_type,
                            'F1-Score': f1,
                            'Precision': precision,
                            'Recall': recall,
                            'Avg Confidence (Correct)': avg_confidence_correct,
                            'Avg Confidence (Wrong)': avg_confidence_wrong,
                            'Support': int(y_true.sum())
                        })
                        
                        # Collect for overall metrics
                        all_predictions.extend(y_pred)
                        all_labels.extend(y_true)
                        all_probabilities.extend(y_prob)
                    
                    # Create metrics dataframe
                    metrics_df = pd.DataFrame(error_type_metrics)
                    
                    # Calculate overall weighted averages
                    supports = metrics_df['Support'].values
                    total_support = supports.sum()
                    
                    if total_support > 0:
                        overall_f1 = np.average(metrics_df['F1-Score'], weights=supports)
                        overall_precision = np.average(metrics_df['Precision'], weights=supports)
                        overall_recall = np.average(metrics_df['Recall'], weights=supports)
                    else:
                        overall_f1 = overall_precision = overall_recall = 0
                    
                    # Overall confidence analysis
                    all_predictions = np.array(all_predictions)
                    all_labels = np.array(all_labels)
                    all_probabilities = np.array(all_probabilities)
                    
                    overall_correct_mask = (all_predictions == all_labels)
                    overall_wrong_mask = (all_predictions != all_labels)
                    
                    overall_avg_confidence_correct = all_probabilities[overall_correct_mask].mean() if overall_correct_mask.sum() > 0 else 0
                    overall_avg_confidence_wrong = all_probabilities[overall_wrong_mask].mean() if overall_wrong_mask.sum() > 0 else 0
                    
                    # Display results
                    st.subheader("📋 Per Error Type Metrics")
                    
                    # Format the dataframe for display
                    display_df = metrics_df.copy()
                    for col in ['F1-Score', 'Precision', 'Recall', 'Avg Confidence (Correct)', 'Avg Confidence (Wrong)']:
                        display_df[col] = display_df[col].round(4)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Overall metrics
                    st.subheader("🎯 Overall Performance (Weighted Average)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall F1-Score", f"{overall_f1:.4f}")
                    with col2:
                        st.metric("Overall Precision", f"{overall_precision:.4f}")
                    with col3:
                        st.metric("Overall Recall", f"{overall_recall:.4f}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Confidence (Correct)", f"{overall_avg_confidence_correct:.4f}")
                    with col2:
                        st.metric("Avg Confidence (Wrong)", f"{overall_avg_confidence_wrong:.4f}")
                    
                    # Visualizations
                    st.subheader("📊 Performance Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<div style='color: white;'><strong>F1-Score by Error Type</strong></div>", unsafe_allow_html=True)
                        f1_chart_df = metrics_df[['Error Type', 'F1-Score']].set_index('Error Type')
                        st.bar_chart(f1_chart_df)
                    
                    with col2:
                        st.markdown("<div style='color: white;'><strong>Confidence Analysis</strong></div>", unsafe_allow_html=True)
                        confidence_df = pd.DataFrame({
                            'Error Type': metrics_df['Error Type'],
                            'Correct Predictions': metrics_df['Avg Confidence (Correct)'],
                            'Wrong Predictions': metrics_df['Avg Confidence (Wrong)']
                        }).set_index('Error Type')
                        st.bar_chart(confidence_df)
                    
                    # Support distribution
                    st.markdown("<div style='color: white;'><strong>Support Distribution (Positive Samples per Error Type)</strong></div>", unsafe_allow_html=True)
                    support_df = metrics_df[['Error Type', 'Support']].set_index('Error Type')
                    st.bar_chart(support_df)
                    
                    # Export functionality
                    st.subheader("📥 Export Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export detailed metrics
                        metrics_csv = metrics_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📊 Download Detailed Metrics",
                            metrics_csv,
                            "multilabel_metrics.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        # Export summary report
                        summary_report = {
                            'overall_f1': float(overall_f1),
                            'overall_precision': float(overall_precision),
                            'overall_recall': float(overall_recall),
                            'overall_avg_confidence_correct': float(overall_avg_confidence_correct),
                            'overall_avg_confidence_wrong': float(overall_avg_confidence_wrong),
                            'total_error_types': len(error_types),
                            'total_records': len(df),
                            'analysis_timestamp': datetime.now().isoformat()
                        }
                        
                        summary_json = json.dumps(summary_report, indent=2).encode('utf-8')
                        st.download_button(
                            "📋 Download Summary Report",
                            summary_json,
                            "multilabel_summary.json",
                            "application/json"
                        )
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("📁 Upload a CSV file to start multilabel classification analysis")
        st.markdown("""
        **Expected CSV Format:**
        - **Metadata columns**: index, premise, hypothesis, label, len_group, doc_id, source, error_types
        - **For each error type X**: 
          - `prob_X`: Probability score (0-1)
          - `pred_X`: Binary prediction (0/1, using 0.5 threshold)
          - `label_X`: Ground truth (0/1)
        
        **Analysis Provided:**
        - 📊 **Per error type**: F1, Precision, Recall scores
        - 🎯 **Confidence analysis**: Average confidence on correct vs wrong predictions
        - 📈 **Overall metrics**: Weighted average across all error types
        - 📋 **Support analysis**: Distribution of positive samples
        - 📥 **Export options**: Detailed metrics and summary reports
        """)

# ===== BINARY CLASSIFIER TAB =====
with tab_binary:
    st.subheader("🎯 Binary Classifier Analysis")
    st.markdown("Analyze binary NLI classification performance (entailment/contradiction).")
    
    # File upload
    binary_file = st.file_uploader(
        "Upload CSV with binary predictions", 
        type=["csv"], 
        key="binary_csv",
        help="CSV from inference_binary_nli.py with true_label, predicted_label, and probability columns"
    )
    
    if binary_file is not None:
        try:
            df_binary = pd.read_csv(binary_file)
            st.success(f"✅ Loaded {len(df_binary)} records")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df_binary))
            with col2:
                st.metric("Total Columns", len(df_binary.columns))
            with col3:
                if 'true_label' in df_binary.columns:
                    st.metric("Unique Labels", df_binary['true_label'].nunique())
            
            # Validate required columns
            required_cols = ['premise', 'hypothesis', 'true_label', 'predicted_label']
            missing_cols = [col for col in required_cols if col not in df_binary.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                # Show sample data
                with st.expander("🔍 Sample Data Preview"):
                    st.dataframe(df_binary.head(), use_container_width=True)
                
                # Get label names
                label_names = sorted(df_binary['true_label'].unique())
                
                # Find probability columns
                prob_cols = [col for col in df_binary.columns if col.startswith('prob_')]
                
                # Perform analysis
                st.subheader("📊 Performance Analysis")
                
                # Calculate metrics
                y_true = df_binary['true_label'].values
                y_pred = df_binary['predicted_label'].values
                
                # Overall metrics
                accuracy = accuracy_score(y_true, y_pred)
                f1_macro = f1_score(y_true, y_pred, average='macro', labels=label_names, zero_division=0)
                f1_micro = f1_score(y_true, y_pred, average='micro', labels=label_names, zero_division=0)
                precision_macro = precision_score(y_true, y_pred, average='macro', labels=label_names, zero_division=0)
                recall_macro = recall_score(y_true, y_pred, average='macro', labels=label_names, zero_division=0)
                
                # Per-class metrics
                precision_per_class = precision_score(y_true, y_pred, average=None, labels=label_names, zero_division=0)
                recall_per_class = recall_score(y_true, y_pred, average=None, labels=label_names, zero_division=0)
                f1_per_class = f1_score(y_true, y_pred, average=None, labels=label_names, zero_division=0)
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=label_names)
                
                # Display overall metrics
                st.subheader("🎯 Overall Performance")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("F1-Score (Macro)", f"{f1_macro:.4f}")
                with col3:
                    st.metric("F1-Score (Micro)", f"{f1_micro:.4f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Precision (Macro)", f"{precision_macro:.4f}")
                with col2:
                    st.metric("Recall (Macro)", f"{recall_macro:.4f}")
                
                # Per-class metrics
                st.subheader("📋 Per-Class Metrics")
                
                per_class_data = []
                for i, label in enumerate(label_names):
                    support = (y_true == label).sum()
                    per_class_data.append({
                        'Label': label,
                        'Precision': precision_per_class[i],
                        'Recall': recall_per_class[i],
                        'F1-Score': f1_per_class[i],
                        'Support': int(support)
                    })
                
                per_class_df = pd.DataFrame(per_class_data)
                
                # Format for display
                display_per_class = per_class_df.copy()
                for col in ['Precision', 'Recall', 'F1-Score']:
                    display_per_class[col] = display_per_class[col].round(4)
                
                st.dataframe(display_per_class, use_container_width=True)
                
                # Confusion Matrix
                st.subheader("🔲 Confusion Matrix")
                
                cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
                st.dataframe(cm_df, use_container_width=True)
                
                # Label distribution
                st.subheader("📊 Label Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div style='color: white;'><strong>True Labels</strong></div>", unsafe_allow_html=True)
                    true_dist = df_binary['true_label'].value_counts()
                    st.bar_chart(true_dist)
                
                with col2:
                    st.markdown("<div style='color: white;'><strong>Predicted Labels</strong></div>", unsafe_allow_html=True)
                    pred_dist = df_binary['predicted_label'].value_counts()
                    st.bar_chart(pred_dist)
                
                # Confidence analysis (if probability columns exist)
                if prob_cols:
                    st.subheader("🎯 Confidence Analysis")
                    
                    # Calculate average confidence for correct vs incorrect predictions
                    correct_mask = (y_true == y_pred)
                    
                    # For each label, get average probability
                    confidence_data = []
                    for label in label_names:
                        prob_col = f'prob_{label}'
                        if prob_col in df_binary.columns:
                            # Correct predictions where this is the true label
                            mask_correct = correct_mask & (y_true == label)
                            # Wrong predictions
                            mask_wrong = ~correct_mask & (y_true == label)
                            
                            avg_conf_correct = df_binary.loc[mask_correct, prob_col].mean() if mask_correct.sum() > 0 else 0
                            avg_conf_wrong = df_binary.loc[mask_wrong, prob_col].mean() if mask_wrong.sum() > 0 else 0
                            
                            confidence_data.append({
                                'Label': label,
                                'Avg Confidence (Correct)': avg_conf_correct,
                                'Avg Confidence (Wrong)': avg_conf_wrong
                            })
                    
                    if confidence_data:
                        conf_df = pd.DataFrame(confidence_data)
                        
                        # Format for display
                        display_conf = conf_df.copy()
                        for col in ['Avg Confidence (Correct)', 'Avg Confidence (Wrong)']:
                            display_conf[col] = display_conf[col].round(4)
                        
                        st.dataframe(display_conf, use_container_width=True)
                        
                        # Visualization
                        st.markdown("<div style='color: white;'><strong>Confidence Comparison</strong></div>", unsafe_allow_html=True)
                        viz_conf = conf_df.set_index('Label')[['Avg Confidence (Correct)', 'Avg Confidence (Wrong)']]
                        st.bar_chart(viz_conf)
                
                # Filter by Confusion Matrix Category
                st.subheader("🔍 Filter Samples by Prediction Category")
                
                # Create filter options based on confusion matrix categories
                filter_options = ["All Samples", "Correct Predictions", "Incorrect Predictions"]
                
                # Add per-label TP/FP/TN/FN options for binary classification
                if len(label_names) == 2:
                    # For binary classification, add TP/FP/TN/FN per label
                    for label in label_names:
                        filter_options.extend([
                            f"True Positive ({label})",
                            f"False Positive ({label})",
                            f"True Negative ({label})",
                            f"False Negative ({label})"
                        ])
                else:
                    # For multi-class, add per-label correct/incorrect
                    for label in label_names:
                        filter_options.extend([
                            f"Correct: {label}",
                            f"Incorrect: {label} (any misprediction)"
                        ])
                
                selected_filter = st.selectbox(
                    "Select prediction category to view:",
                    options=filter_options,
                    key="binary_filter_category"
                )
                
                # Apply filter
                filtered_df = df_binary.copy()
                
                if selected_filter == "Correct Predictions":
                    filtered_df = df_binary[y_true == y_pred]
                elif selected_filter == "Incorrect Predictions":
                    filtered_df = df_binary[y_true != y_pred]
                elif selected_filter.startswith("True Positive"):
                    # Extract label name
                    label = selected_filter.split("(")[1].rstrip(")")
                    # TP: true label = label AND predicted label = label
                    filtered_df = df_binary[(y_true == label) & (y_pred == label)]
                elif selected_filter.startswith("False Positive"):
                    label = selected_filter.split("(")[1].rstrip(")")
                    # FP: true label != label AND predicted label = label
                    filtered_df = df_binary[(y_true != label) & (y_pred == label)]
                elif selected_filter.startswith("True Negative"):
                    label = selected_filter.split("(")[1].rstrip(")")
                    # TN: true label != label AND predicted label != label
                    filtered_df = df_binary[(y_true != label) & (y_pred != label)]
                elif selected_filter.startswith("False Negative"):
                    label = selected_filter.split("(")[1].rstrip(")")
                    # FN: true label = label AND predicted label != label
                    filtered_df = df_binary[(y_true == label) & (y_pred != label)]
                elif selected_filter.startswith("Correct:"):
                    label = selected_filter.split(": ")[1]
                    # Correct predictions where true label = label
                    filtered_df = df_binary[(y_true == label) & (y_pred == label)]
                elif selected_filter.startswith("Incorrect:"):
                    label = selected_filter.split(": ")[1].split(" (")[0]
                    # Incorrect predictions where true label = label but predicted != label
                    filtered_df = df_binary[(y_true == label) & (y_pred != label)]
                
                # Display filtered results
                st.markdown(f"<div style='color: white;'><strong>Showing: {selected_filter}</strong></div>", unsafe_allow_html=True)
                
                if len(filtered_df) > 0:
                    # Calculate stats for filtered subset
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Filtered Records", len(filtered_df))
                    
                    # Calculate average confidence scores
                    if prob_cols:
                        # Get all probability columns
                        all_probs = filtered_df[prob_cols].values
                        # Get max probability per row (confidence of prediction)
                        max_probs = all_probs.max(axis=1)
                        avg_confidence = max_probs.mean()
                        
                        with col2:
                            st.metric("Avg Confidence Score", f"{avg_confidence:.4f}")
                    
                    # Calculate text lengths
                    if 'premise' in filtered_df.columns:
                        premise_lengths = filtered_df['premise'].fillna('').str.split().str.len()
                        avg_premise_len = premise_lengths.mean()
                        
                        with col3:
                            st.metric("Avg Premise Length", f"{avg_premise_len:.1f} words")
                    
                    if 'hypothesis' in filtered_df.columns:
                        hypothesis_lengths = filtered_df['hypothesis'].fillna('').str.split().str.len()
                        avg_hypothesis_len = hypothesis_lengths.mean()
                        
                        with col4:
                            st.metric("Avg Hypothesis Length", f"{avg_hypothesis_len:.1f} words")
                    
                    # View options
                    view_all = st.checkbox("📄 Show all filtered records (not just sample)", key="view_all_filtered")
                    
                    # Show data
                    display_cols = ['premise', 'hypothesis', 'true_label', 'predicted_label']
                    if prob_cols:
                        display_cols.extend(prob_cols)
                    
                    if view_all:
                        st.dataframe(filtered_df[display_cols], use_container_width=True)
                    else:
                        with st.expander(f"📋 Sample Data ({min(10, len(filtered_df))} records)"):
                            st.dataframe(filtered_df[display_cols].head(10), use_container_width=True)
                    
                    # Download filtered data
                    filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        f"📥 Download Filtered Data ({len(filtered_df)} records)",
                        filtered_csv,
                        f"filtered_{selected_filter.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}.csv",
                        "text/csv",
                        key="download_filtered_binary"
                    )
                else:
                    st.warning("⚠️ No records match the selected filter criteria")
                
                # Export functionality
                st.subheader("📥 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export per-class metrics
                    per_class_csv = per_class_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📊 Download Per-Class Metrics",
                        per_class_csv,
                        "binary_per_class_metrics.csv",
                        "text/csv"
                    )
                
                with col2:
                    # Export summary report
                    binary_summary = {
                        'accuracy': float(accuracy),
                        'f1_macro': float(f1_macro),
                        'f1_micro': float(f1_micro),
                        'precision_macro': float(precision_macro),
                        'recall_macro': float(recall_macro),
                        'labels': label_names,
                        'confusion_matrix': cm.tolist(),
                        'total_records': len(df_binary),
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                    
                    binary_json = json.dumps(binary_summary, indent=2).encode('utf-8')
                    st.download_button(
                        "📋 Download Summary Report",
                        binary_json,
                        "binary_summary.json",
                        "application/json"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("📁 Upload a CSV file to start binary classification analysis")
        st.markdown("""
        **Expected CSV Format:**
        - **Required columns**: premise, hypothesis, true_label, predicted_label
        - **Optional columns**: prob_entailment, prob_contradiction, prob_neutral (probability scores)
        
        **Analysis Provided:**
        - 🎯 **Overall metrics**: Accuracy, F1-Score (macro/micro), Precision, Recall
        - 📋 **Per-class metrics**: Precision, Recall, F1-Score, Support for each label
        - 🔲 **Confusion matrix**: Visual representation of predictions vs ground truth
        - 📊 **Label distribution**: True vs predicted label distribution
        - 🎯 **Confidence analysis**: Average confidence for correct vs incorrect predictions (if probabilities available)
        - 📥 **Export options**: Detailed metrics and summary reports
        """)
