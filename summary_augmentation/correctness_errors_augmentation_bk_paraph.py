# -*- coding: utf-8 -*-
import concurrent.futures
import threading
import datetime
import pandas as pd
import logging
import sys
import os
from typing import List, Dict, Any
import logging
from typing import List

import datetime

from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict

# Configure the logger (only once, typically at the top of your script or entry point)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json
import summary_augmentation.correctness_errors_examples as correctness_errors_examples
from openai import OpenAI
# Define OpenAI API credentials
#import google.generativeai as genai
from langchain_openai import ChatOpenAI

import threading
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from summary_augmentation.correctness_errors_llm_helper_functions import verify_transformed_summary_grammar, choose_suitable_claim, summary_negative_augmentation, summary_halluzaition_negative_augmentation, back_translation_and_paraphrasing



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
    # configuration
    # NEW: let user pick slice
    start_row = 40
    end_row   = 55
    input_csv_path  = r"C:\Users\amita\Downloads\correctness metric data\b2_a2_not_in_mafat_800_article_150_summary.csv"
    timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = rf"C:\Users\amita\Downloads\bt_paraphrased_{timestamp}_{start_row}_{end_row}.json"
    output_csv_path  = rf"C:\Users\amita\Downloads\bt_paraphrased_{timestamp}_{start_row}_{end_row}.csv"


    records: List[Dict] = []

    # read the CSV
    with open(input_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < start_row or idx >= end_row:
                continue

            premise    = row['premise']
            hypothesis = row['hypothesis']


            # call your transformer
            result = back_translation_and_paraphrasing(hypothesis)
            if not result:
                continue

            # two new entailment records, both under 'hypothesis'
            records.append({
                'premise': premise,
                'hypothesis': result.summary_paraphrasing,
                'label': 'entailment',
                'group_len': row['len_group'],
                'doc_id': row['doc_id'],
                'source': row['source']
            })
            records.append({
                'premise': premise,
                'hypothesis': result.summary_back_translation,
                'label': 'entailment',
                'group_len': row['len_group'],
                'doc_id': row['doc_id'],
                'source': row['source']

            })

    # write out to JSON
    with open(output_json_path, 'w', encoding='utf-8') as out_json:
        json.dump(records, out_json, ensure_ascii=False, indent=2)

    # write out to CSV
    if records:
        fieldnames = list(records[0].keys())
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    print(f"Wrote {len(records)} records to:\n  JSON: {output_json_path}\n  CSV:  {output_csv_path}")

if __name__ == "__main__":
    main()

# def main():
#     # input_csv_path = r"/sise/home/avramit/dataset_augmentation/output_mafat_less_500_test_new.csv"
#     #input_csv_path = r"/sise/home/avramit/mfact-summ/mfact/datasets/heb/mafat_800_article_150_summary.csv"
#     input_csv_path = r"C:\Users\amita\Downloads\mafat_800_article_150_summary.csv"
#     output_csv_path = r"/sise/home/avramit/dataset_augmentation/output_mafat_less_500_test_negative.csv"
#     num_rows_to_transform = 20
#     transformation_types_1 = [CorrectnessErrorTypes.ENTITY_PERSON_SWAP, CorrectnessErrorTypes.ENTITY_ORGANIZATION_SWAP,
#                               CorrectnessErrorTypes.ENTITY_LOCATION_SWAP,CorrectnessErrorTypes.ENTITY_DATE_SWAP, CorrectnessErrorTypes.ENTITY_EVENT_SWAP,
#                               CorrectnessErrorTypes.ENTITY_TITLE_SWAP]
#
#     # Load the CSV data
#     df = pd.read_csv(input_csv_path)
#     # Take the first n rows to transform
#     if num_rows_to_transform < len(df):
#         rows_to_transform = df.sample(n=num_rows_to_transform)
#     else:
#         rows_to_transform = df  # Use all rows if requested more than available
#     # Create a list to store new rows
#     new_rows = []
#
#     # Process each selected row
#     for _, row in rows_to_transform.iterrows():
#         # summary = row['hypothesis']
#         # premise = row['premise']
#         premise = row['hypothesis']
#         summary = row['premise']
#         new_sum=correctness_errors_transformation_workflow(summary=summary, source_article=premise, transformation_types=transformation_types_1)
#
#     # Create DataFrame with new rows
#     new_df = pd.DataFrame(new_rows)
#
#     # Append new rows to original DataFrame and save
#     result_df = pd.concat([df, new_df], ignore_index=True)
#     result_df.to_csv(output_csv_path, index=False)
#
#     print(f"Successfully processed {len(new_rows)} rows and saved to {output_csv_path}")
#     return result_df



# Example of how to call the main function:
# if __name__ == "__main__":
#     print("hello")
#     main()
