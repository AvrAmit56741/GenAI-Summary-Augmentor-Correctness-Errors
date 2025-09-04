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
# Cnfigure he logger (only once, typically at the top of your script or entry point)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import csv
import json
import datetime
from typing import List, Dict

import threading
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from summary_augmentation.correctness_errors_llm_helper_functions import correctness_error_finder



def main():
    # configuration
    start_row = 91
    end_row   = 110
    #input_jsonl_path = r"C:\Users\amita\Desktop\Thesis\Correctness\data\summarization-7-heb.jsonl\summarization-7-heb.jsonl"
    input_jsonl_path = r"C:\Users\amita\Desktop\Thesis\summaries_dicta_30_200.jsonl"
    timestamp        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = rf"C:\Users\amita\Downloads\finder_{timestamp}_{start_row}_{end_row}_dicta2.0_summaries_gemini_finder.json"
    output_csv_path  = rf"C:\Users\amita\Downloads\finder_{timestamp}_{start_row}_{end_row}_dicta2.0_summaries_gemini_finder.csv"

    records: List[Dict] = []

    # read the JSONL
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):

            if idx < start_row or idx >= end_row:
                continue

            obj = json.loads(line)

            print(f"analyze line {idx}")
            # only silver
            # if obj.get('metadata', {}).get('type') != "silver":
            #     continue
            # if obj.get('metadata', {}).get('source') == "Israel_Hayom":
            #     continue
            premise    = obj.get('text_raw', '').strip()
            #hypothesis = obj.get('metadata', {}).get('ai_summary', '').strip()
            hypothesis = obj.get('metadata', {}).get('generative_summaries', [])[-1].get('generative_summary', '').strip()

            # call your transformer
            result = correctness_error_finder(premise, hypothesis)
            # skip if no errors
            if not result or not result.summary_article_heb_correctness_errors:
                continue

            # one row per error
            for err in result.summary_article_heb_correctness_errors:
                records.append({
                    'premise': premise,
                    "ai_summary": obj.get('metadata', {}).get('ai_summary', '').strip(),
                    "generative_summary": obj.get('metadata', {}).get('generative_summaries', [])[-1].get('generative_summary', '').strip(),
                    'correctness_error_name':        err.correctness_error_name,
                    'correctness_error_reasoning':   err.correctness_error_reasoning,
                    'mrl_unique_hebrew_features':    err.mrl_unique_hebrew_features,
                    'evidence_from_summary':         err.evidence_from_summary,
                    'why_its_factual_error': err.why_its_factual_error,
                    'evidence_from_source_article':  err.evidence_from_source_article,
                    'severity': err.severity,
                    'doc_id':      obj['metadata'].get('doc_id'),
                    'source':      obj['metadata'].get('source'),
                    'label':       'entailment',
                })

    # write out to JSON
    with open(output_json_path, 'w', encoding='utf-8') as out_json:
        json.dump(records, out_json, ensure_ascii=False, indent=2)

    # write out to CSV
    if records:
        import csv
        fieldnames = list(records[0].keys())
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    print(f"Wrote {len(records)} records to:\n  JSON: {output_json_path}\n  CSV:  {output_csv_path}")

if __name__ == "__main__":
    main()