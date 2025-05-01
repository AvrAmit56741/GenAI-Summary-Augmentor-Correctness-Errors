#!/usr/bin/env python3
import json
import csv
import sys


def main():
    input_file = "C:\\Users\\amita\\Desktop\\Thesis\\Correctness\\data\\summarization-7-heb.jsonl\\summarization-7-heb.jsonl"
    output_file = "summarization-7-heb.csv"
    num_objects = 100  # Hardcoded number of objects to process

    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_objects:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {i + 1}")
                continue
            # Extract fields from JSON object
            premise = obj.get("text_raw", "")
            hypothesis = obj.get("summary", "")
            label = "entailment"
            rows.append([i, premise, hypothesis, label])

    # Write to CSV file with header row
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "premise", "hypothesis", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    main()
