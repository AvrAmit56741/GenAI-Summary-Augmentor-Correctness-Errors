# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import datetime
from typing import List, Dict
import json
from summary_augmentation.correctness_errors_llm_helper_functions import correctness_error_finder



def main():
    # configuration
    start_row = 85
    end_row   = 95
    input_jsonl_path = r"C:\Users\amita\Desktop\Thesis\summaries_ollama_qwen3-14b_20250923_184510_start_row_150_end_row_250.jsonl"
    timestamp        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = rf"C:\Users\amita\Downloads\finder_{timestamp}_{start_row}_{end_row}_ollama_gemma3-12b_summaries_gpt_finder.json"
    output_csv_path  = rf"C:\Users\amita\Downloads\finder_{timestamp}_{start_row}_{end_row}_ollama_gemma3-12b_summaries_gpt_finder.csv"

    records: List[Dict] = []

    # read the JSONL
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):

            if idx < start_row or idx >= end_row:
                continue

            obj = json.loads(line)

            print(f"analyze line {idx}")
            premise    = obj.get('text_raw', '').strip()
            hypothesis = obj.get('metadata', {}).get('generative_summaries', [])[-1].get('generative_summary', '').strip()
            result = correctness_error_finder(premise, hypothesis)
            if not result or not result.summary_article_heb_correctness_errors:
                continue
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

    with open(output_json_path, 'w', encoding='utf-8') as out_json:
        json.dump(records, out_json, ensure_ascii=False, indent=2)

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