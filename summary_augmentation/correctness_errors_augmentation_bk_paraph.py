# -*- coding: utf-8 -*-
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import threading
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from summary_augmentation.correctness_errors_llm_helper_functions import verify_transformed_summary_grammar, choose_suitable_claim, summary_negative_augmentation, summary_halluzaition_negative_augmentation, back_translation_and_paraphrasing

# Import similarity metrics
try:
    from bert_score import score as bertscore_calc
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("bert_score not installed. BERTScore verification will be disabled.")

try:
    import sacrebleu
    CHRF_AVAILABLE = True
except ImportError:
    CHRF_AVAILABLE = False
    logger.warning("sacrebleu not installed. ChrF verification will be disabled.")


def calculate_chrf_similarity(reference, hypothesis):
    """Calculate ChrF score between reference and hypothesis."""
    if not CHRF_AVAILABLE:
        return None
    try:
        chrf = sacrebleu.sentence_chrf(hypothesis, [reference])
        return chrf.score / 100.0  # Convert to 0-1 range
    except Exception as e:
        logger.warning(f"ChrF calculation failed: {e}")
        return None


def calculate_bertscore_similarity(reference, hypothesis, model_type):
    """Calculate BERTScore F1 between reference and hypothesis."""
    if not BERTSCORE_AVAILABLE:
        return None
    try:
        # For Hebrew, use lang='he' which will use appropriate multilingual model
        # Or explicitly set model_type if needed
        if model_type:
            P, R, F1 = bertscore_calc([hypothesis], [reference], model_type=model_type, verbose=False)
        else:
            P, R, F1 = bertscore_calc([hypothesis], [reference], lang='he', verbose=False)
        return F1.item()
    except Exception as e:
        logger.warning(f"BERTScore calculation failed: {e}")
        logger.debug(f"Attempted model: {model_type}, reference: {reference[:50]}, hypothesis: {hypothesis[:50]}")
        return None


def verify_similarity(original, generated, similarity_metric, chrf_threshold, bertscore_threshold, bertscore_model):
    """
    Verify similarity between original and generated text.
    Returns: (passed, scores_dict)
    """
    scores = {}
    passed = True
    
    if similarity_metric is None or similarity_metric == "none":
        return True, scores
    
    # Calculate ChrF if needed
    if similarity_metric in ["chrf", "both"]:
        chrf_score = calculate_chrf_similarity(original, generated)
        if chrf_score is not None:
            scores['chrf'] = chrf_score
            if chrf_score < chrf_threshold:
                passed = False
                logger.debug(f"ChrF score {chrf_score:.3f} below threshold {chrf_threshold}")
    
    # Calculate BERTScore if needed
    if similarity_metric in ["bert", "both"]:
        bert_score = calculate_bertscore_similarity(original, generated, bertscore_model)
        if bert_score is not None:
            scores['bertscore'] = bert_score
            if bert_score < bertscore_threshold:
                passed = False
                logger.debug(f"BERTScore {bert_score:.3f} below threshold {bertscore_threshold}")
    
    return passed, scores


def process_row(row, transform_types):
    """Worker function to process a single row."""
    thread_id = threading.current_thread().name
    try:
        # note: premise/hypothesis as in original code
        source = row['hypothesis']
        gold = row['premise']

        corrupted_list = correctness_errors_transformation_workflow(
            summary=gold,
            source_article=source,
            transformation_types=transform_types
        )

        logger.info(f"Thread {thread_id}: generated {len(corrupted_list)} corrupted summaries")
        return {
            'source_article': source,
            'gold_summary': gold,
            'correctness_corrupted_summaries': corrupted_list
        }
    except Exception as e:
        logger.error(f"Thread {thread_id}: Error processing row: {str(e)}")
        return None


import csv
import json
import datetime
from typing import List, Dict

def main():
    # Configuration
    start_row = 351
    end_row = 450
    filter_by_label = None
    max_workers = 5

    similarity_metric = "both"
    chrf_threshold = 0.64  # ChrF score threshold (0.0-1.0)
    bertscore_threshold = 0.85  # BERTScore F1 threshold (0.0-1.0)
    bertscore_model = None  # Use default multilingual BERT for Hebrew (or specify: "bert-base-multilingual-cased")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_type = "validation"
    input_csv_path = rf"C:\Users\amita\Downloads\correctness metric data\All dataset split\mafat\{dataset_type}\{dataset_type}.csv"
    output_json = rf"C:\Users\amita\Downloads\correctness metric data\All dataset split\mafat\bk_pr\{dataset_type}\mafat_bk_pr_{dataset_type}_{timestamp}_start_row_{start_row}_end_row_{end_row}.json"
    output_csv = rf"C:\Users\amita\Downloads\correctness metric data\All dataset split\mafat\bk_pr\{dataset_type}\mafat_bk_pr_{dataset_type}_{timestamp}_start_row_{start_row}_end_row_{end_row}.csv"

    logger.info(f"Starting processing with {max_workers} worker threads")
    logger.info(f"Processing rows {start_row} to {end_row - 1}")
    if filter_by_label:
        logger.info(f"Filtering samples by label: {filter_by_label}")

    logger.info(f"Quality verification: metric={similarity_metric}, chrf_threshold={chrf_threshold}, bertscore_threshold={bertscore_threshold}")
    logger.info(f"Libraries available - ChrF: {CHRF_AVAILABLE}, BERTScore: {BERTSCORE_AVAILABLE}")
    df = pd.read_csv(input_csv_path)

    if filter_by_label:
        original_count = len(df)
        df = df[df['label'] == filter_by_label].reset_index(drop=True)
        filtered_count = len(df)
        logger.info(f"Applied label filter '{filter_by_label}': {original_count} → {filtered_count} samples")
    slice_df = df.iloc[start_row:end_row]
    logger.info(f"Processing {len(slice_df)} samples")

    records: List[Dict] = []
    failed_samples = []
    start_time = datetime.datetime.now()
    records_lock = threading.Lock()
    failed_lock = threading.Lock()

    def process_single_sample(row):
        """Process a single sample with back-translation and paraphrasing."""
        try:
            idx = row.name
            premise = row['premise']
            hypothesis = row['hypothesis']
            
            logger.info(f"Processing row {idx}: doc_id={row.get('doc_id', 'unknown')}")
            result = back_translation_and_paraphrasing(hypothesis)
            if not result:
                logger.warning(f"Skipping row {idx}: transformation failed")
                return None
            if not result.summary_paraphrasing or not result.summary_paraphrasing.strip():
                logger.warning(f"Skipping row {idx}: paraphrasing returned empty text")
                return None
            
            if not result.summary_back_translation or not result.summary_back_translation.strip():
                logger.warning(f"Skipping row {idx}: back_translation returned empty text")
                return None
            paraphrase_passed, paraphrase_scores = verify_similarity(
                hypothesis, result.summary_paraphrasing,
                similarity_metric, chrf_threshold, bertscore_threshold, bertscore_model
            )
            
            if not paraphrase_passed:
                logger.warning(f"Skipping row {idx}: paraphrasing failed similarity check - {paraphrase_scores}")
                with failed_lock:
                    failed_samples.append({
                        'doc_id': row.get('doc_id', 'unknown'),
                        'method': 'paraphrasing',
                        'reason': 'similarity_check_failed',
                        'scores': paraphrase_scores
                    })
                return None
            backtrans_passed, backtrans_scores = verify_similarity(
                hypothesis, result.summary_back_translation,
                similarity_metric, chrf_threshold, bertscore_threshold, bertscore_model
            )
            
            if not backtrans_passed:
                logger.warning(f"Skipping row {idx}: back_translation failed similarity check - {backtrans_scores}")
                with failed_lock:
                    failed_samples.append({
                        'doc_id': row.get('doc_id', 'unknown'),
                        'method': 'back_translation',
                        'reason': 'similarity_check_failed',
                        'scores': backtrans_scores
                    })
                return None

            logger.info(f"✓ Similarity checks passed - Paraphrase: {paraphrase_scores}, BackTrans: {backtrans_scores}")
            sample_records = [
                {
                    'premise': premise,
                    'hypothesis': result.summary_paraphrasing,
                    'label': 'entailment',
                    'len_group': row['len_group'],
                    'doc_id': row['doc_id'],
                    'source': row['source'],
                    'error_types': '',
                    'method': 'paraphrasing',
                    'original_hypothesis': hypothesis
                },
                {
                    'premise': premise,
                    'hypothesis': result.summary_back_translation,
                    'label': 'entailment',
                    'len_group': row['len_group'],
                    'doc_id': row['doc_id'],
                    'source': row['source'],
                    'error_types': '',
                    'method': 'back_translation',
                    'original_hypothesis': hypothesis
                }
            ]
            
            logger.info(f"✓ Row {idx} completed: generated 2 samples")
            return sample_records
            
        except Exception as e:
            logger.error(f"Error processing row {row.name}: {e}")
            return None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_sample, row) for idx, row in slice_df.iterrows()]
        
        for future in futures:
            result = future.result()
            if result:
                with records_lock:
                    records.extend(result)
    with open(output_json, 'w', encoding='utf-8') as out_json:
        json.dump(records, out_json, ensure_ascii=False, indent=2)

    # Write out to CSV
    if records:
        fieldnames = ['premise', 'hypothesis', 'label', 'len_group', 'doc_id', 'source', 'error_types', 'method', 'original_hypothesis']
        with open(output_csv, 'w', newline='', encoding='utf-8') as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    output_summary = output_csv.replace('.csv', '_summary.txt')
    
    total_attempted = len(slice_df)
    total_generated = len(records)
    total_failed = len(failed_samples)
    success_rate = (total_generated / (total_attempted * 2) * 100) if total_attempted > 0 else 0
    
    summary_content = f"""
================================================================================
DATA AUGMENTATION SUMMARY - Back-Translation & Paraphrasing
================================================================================

Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)

INPUT/OUTPUT FILES:
  Input:  {input_csv_path}
  Output CSV:  {output_csv}
  Output JSON: {output_json}
  
PROCESSING SCOPE:
  Rows processed: {start_row} to {end_row-1} ({total_attempted} samples)
  Label filter: {filter_by_label if filter_by_label else 'None (all labels)'}

QUALITY VERIFICATION SETTINGS:
  Similarity metric: {similarity_metric}
  ChrF threshold: {chrf_threshold}
  BERTScore threshold: {bertscore_threshold}
  BERTScore model: {bertscore_model if bertscore_model else 'Default (multilingual BERT)'}
  Libraries available - ChrF: {CHRF_AVAILABLE}, BERTScore: {BERTSCORE_AVAILABLE}

RESULTS:
  Total samples attempted: {total_attempted}
  Potential outputs (2 per sample): {total_attempted * 2}
  Successfully generated: {total_generated}
  Failed validation: {total_failed}
  Success rate: {success_rate:.1f}%
  
  Breakdown:
    - Paraphrase samples: {sum(1 for r in records if r['method'] == 'paraphrasing')}
    - Back-translation samples: {sum(1 for r in records if r['method'] == 'back_translation')}

FAILED SAMPLES DETAILS:
"""
    
    if failed_samples:
        for i, fail in enumerate(failed_samples, 1):
            summary_content += f"\n  {i}. Doc ID: {fail['doc_id']}"
            summary_content += f"\n     Method: {fail['method']}"
            summary_content += f"\n     Reason: {fail['reason']}"
            summary_content += f"\n     Scores: {fail['scores']}"
    else:
        summary_content += "\n  No failed samples (all passed validation)"
    
    summary_content += "\n\n" + "="*80 + "\n"
    with open(output_summary, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    logger.info(f"✓ Wrote {len(records)} records to:")
    logger.info(f"  JSON: {output_json}")
    logger.info(f"  CSV:  {output_csv}")
    logger.info(f"  Summary: {output_summary}")
    print(f"\n✅ Processing complete: {len(records)} records saved")
    print(f"⏱️  Duration: {duration:.2f}s | Success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    main()

