import json
import os
import csv
from pathlib import Path
from tqdm import tqdm

print("v2 - Strategy 3: ALWAYS create entailment (original->original) for every claim")
# ========================
# CONFIGURATION
# ========================
INPUT_MODE = "directory"  # Options: "file" or "directory"
INPUT_PATH = "/home/avramit/classifier/01122025/datasets/classifiers/mafat_json/"  # Single JSON file or directory with train/val/test files
OUTPUT_FOLDER = None  # Will be set automatically as INPUT_PATH/parse_orgclaim_augclaim
print("update!!")

def parse_json_to_csv(json_path, output_csv_path):
    """
    Parse augmentation JSON and create CSV with:
    - premise: original_claim
    - hypothesis: transformed_claim (or original_claim for entailment)
    - error_types: comma-separated error types (empty for entailment)
    - label: "contradiction" or "entailment"
    """
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    stats = {"entailment": 0, "contradiction": 0, "error_types": {}}

    print("Processing claims...")
    for record in tqdm(data):
        for corrupted_obj in record.get("correctness_corrupted_summaries", []):
            transformation_list = corrupted_obj.get("transformation_list", [])

            for transform in transformation_list:
                original_claim = transform["original_claim"]
                
                # ALWAYS add entailment sample (original → original)
                rows.append({
                    "premise": original_claim,
                    "hypothesis": original_claim,
                    "error_types": "",
                    "label": "entailment"
                })
                stats["entailment"] += 1
                
                # Add contradiction ONLY if claim was corrupted
                if transform.get("applied_transformations"):
                    transformed_claim = transform["transformed_claim"]
                    
                    # Collect all error types for this claim
                    error_types = [t["type"] for t in transform["applied_transformations"]]
                    error_types_str = ",".join(error_types)
                    
                    # Track error type stats
                    for err_type in error_types:
                        stats["error_types"][err_type] = stats["error_types"].get(err_type, 0) + 1
                    
                    rows.append({
                        "premise": original_claim,
                        "hypothesis": transformed_claim,
                        "error_types": error_types_str,
                        "label": "contradiction"
                    })
                    stats["contradiction"] += 1
    
    # Write CSV
    print(f"Writing CSV to {output_csv_path}...")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Clean rows - ensure no None values
    cleaned_rows = []
    for row in rows:
        cleaned_row = {
            "premise": str(row.get("premise", "")).replace('\x00', ''),
            "hypothesis": str(row.get("hypothesis", "")).replace('\x00', ''),
            "error_types": str(row.get("error_types", "")),
            "label": str(row.get("label", "contradiction"))
        }
        cleaned_rows.append(cleaned_row)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["premise", "hypothesis", "error_types", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, escapechar='\\')
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    print(f"✅ Created {len(rows)} samples in {output_csv_path}\n")
    
    # Add sample rows to stats for verification
    stats["samples"] = {
        "entailment": next((r for r in cleaned_rows if r["label"] == "entailment"), None),
        "contradiction": next((r for r in cleaned_rows if r["label"] == "contradiction"), None)
    }
    
    return len(rows), stats


def display_stats(all_stats):
    """Display aggregated statistics and samples"""
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    
    total_entailment = sum(s["entailment"] for s in all_stats.values())
    total_contradiction = sum(s["contradiction"] for s in all_stats.values())
    total_samples = total_entailment + total_contradiction
    
    print(f"\n📈 Label Distribution:")
    print(f"  Entailment:    {total_entailment:,} ({100*total_entailment/total_samples:.1f}%)")
    print(f"  Contradiction: {total_contradiction:,} ({100*total_contradiction/total_samples:.1f}%)")
    print(f"  TOTAL:         {total_samples:,}")
    print(f"  Ratio (E:C):   {total_entailment/total_contradiction:.2f}:1")
    
    # Aggregate error types
    all_error_types = {}
    for stats in all_stats.values():
        for err_type, count in stats["error_types"].items():
            all_error_types[err_type] = all_error_types.get(err_type, 0) + count
    
    print(f"\n🔍 Error Types Distribution ({len(all_error_types)} unique types):")
    sorted_errors = sorted(all_error_types.items(), key=lambda x: x[1], reverse=True)
    for err_type, count in sorted_errors[:10]:  # Show top 10
        print(f"  {err_type:30s} {count:5,} ({100*count/total_contradiction:.1f}%)")
    if len(sorted_errors) > 10:
        print(f"  ... and {len(sorted_errors)-10} more error types")
    
    # Show samples
    print(f"\n📝 Sample Verification:")
    for split_name, stats in all_stats.items():
        if stats["samples"]["entailment"] and stats["samples"]["contradiction"]:
            print(f"\n  [{split_name.upper()}]")
            
            ent = stats["samples"]["entailment"]
            print(f"  ✅ ENTAILMENT sample:")
            print(f"     Premise:    {ent['premise'][:80]}...")
            print(f"     Hypothesis: {ent['hypothesis'][:80]}...")
            print(f"     Error types: '{ent['error_types']}'")
            
            contra = stats["samples"]["contradiction"]
            print(f"  ❌ CONTRADICTION sample:")
            print(f"     Premise:    {contra['premise'][:80]}...")
            print(f"     Hypothesis: {contra['hypothesis'][:80]}...")
            print(f"     Error types: '{contra['error_types']}'")
            break  # Only show first split to save space
    
    print("\n" + "="*60)


def main():
    # Set output folder inside input path
    global OUTPUT_FOLDER
    if OUTPUT_FOLDER is None:
        if INPUT_MODE == "directory":
            OUTPUT_FOLDER = os.path.join(INPUT_PATH, "parse_orgclaim_augclaim")
        else:
            # For file mode, use parent directory of the file
            OUTPUT_FOLDER = os.path.join(os.path.dirname(INPUT_PATH), "parse_orgclaim_augclaim")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    all_stats = {}
    
    if INPUT_MODE == "file":
        # Single file mode
        output_csv = os.path.join(OUTPUT_FOLDER, "parsed_data.csv")
        count, stats = parse_json_to_csv(INPUT_PATH, output_csv)
        all_stats["data"] = stats
        
    elif INPUT_MODE == "directory":
        # Directory mode - process train/val/test files
        input_dir = Path(INPUT_PATH)
        total_samples = 0
        
        for json_file in input_dir.glob("*.json"):
            filename = json_file.stem.lower()
            
            # Determine output name based on train/val/test in filename
            if "train" in filename:
                output_name = "train.csv"
                split_name = "train"
            elif "val" in filename or "validation" in filename:
                output_name = "val.csv"
                split_name = "val"
            elif "test" in filename:
                output_name = "test.csv"
                split_name = "test"
            else:
                output_name = f"{filename}.csv"
                split_name = filename
            
            output_csv = os.path.join(OUTPUT_FOLDER, output_name)
            count, stats = parse_json_to_csv(str(json_file), output_csv)
            total_samples += count
            all_stats[split_name] = stats
        
        print(f"\n🎉 Total samples processed: {total_samples}")
    
    else:
        raise ValueError(f"Invalid INPUT_MODE: {INPUT_MODE}. Use 'file' or 'directory'")
    
    # Display aggregated statistics
    display_stats(all_stats)


if __name__ == "__main__":
    print("="*50)
    print("AUGMENTATION JSON TO CSV PARSER")
    print("="*50)
    print(f"Input mode: {INPUT_MODE}")
    print(f"Input path: {INPUT_PATH}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("="*50 + "\n")
    
    main()