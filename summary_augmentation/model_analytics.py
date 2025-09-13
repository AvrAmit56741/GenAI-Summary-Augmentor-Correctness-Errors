import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def model_analytics_page():
    """
    Model Analytics page - separate from main UI
    Provides comprehensive model performance analysis and visualization
    """
    st.title("🔬 Model Analytics")
    st.markdown('<style>div.stMarkdown p, div.stMarkdown span, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: white; }</style>', unsafe_allow_html=True)
    
    st.markdown("Advanced analytics and performance evaluation for Hebrew NLI models.")
    
    # Create internal tabs for Model Analytics
    tab_multilabel, tab_binary = st.tabs([
        "🏷️ Multilabel Classifier", "🎯 Binary Classifier"
    ])
    
    with tab_multilabel:
        multilabel_classifier_analysis()
    
    with tab_binary:
        binary_classifier_analysis()

def multilabel_classifier_analysis():
    """Multilabel Classifier Analysis tab content"""
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
                return
            
            # Extract error types and validate triplets
            if not error_types:
                st.error("❌ No error type columns found. Expected columns starting with 'prob_'")
                return
            
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
                return
            
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
                    'Support': y_true.sum()  # Number of positive samples
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
                    'overall_f1': overall_f1,
                    'overall_precision': overall_precision,
                    'overall_recall': overall_recall,
                    'overall_avg_confidence_correct': overall_avg_confidence_correct,
                    'overall_avg_confidence_wrong': overall_avg_confidence_wrong,
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

def binary_classifier_analysis():
    """Binary Classifier Analysis tab content"""
    st.subheader("🎯 Binary Classifier Analysis")
    st.info("🚧 This section will contain binary classification analysis functionality.")
    
    st.markdown("**Planned Features:**")
    st.markdown("- Binary classification performance metrics")
    st.markdown("- ROC curve analysis")
    st.markdown("- Precision-Recall curves")
    st.markdown("- Threshold optimization")
    st.markdown("- Confusion matrix visualization")

# Test function to run the page independently
if __name__ == "__main__":
    st.set_page_config(page_title="Model Analytics", layout="wide")
    model_analytics_page() 