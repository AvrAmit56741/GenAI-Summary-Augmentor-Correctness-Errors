import json
import os
import csv
from pathlib import Path
from tqdm import tqdm

# ========================
# CONFIGURATION
# ========================
INPUT_MODE = "directory"  # Options: "file" or "directory"
INPUT_PATH = "/home/avramit/llm_finetune/llm_evidence_big_dataset/"  # Single JSON file or directory with train/val/test files
OUTPUT_FOLDER = None  # Will be set automatically as INPUT_PATH/parse_orgclaim_augclaim
print("update!!")

def parse_json_to_csv(json_path, output_csv_path):
    """
    Parse augmentation JSON and create CSV with:
    - premise: original_claim
    - hypothesis: transformed_claim
    - error_types: comma-separated error types
    - label: always "contradiction"
    """
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []

    print("Processing claims...")
    for record in tqdm(data):
        for corrupted_obj in record.get("correctness_corrupted_summaries", []):
            transformation_list = corrupted_obj.get("transformation_list", [])

            for transform in transformation_list:
                if not transform.get("applied_transformations"):
                    continue  # Skip claims with no errors

                original_claim = transform["original_claim"]
                transformed_claim = transform["transformed_claim"]
                
                # Collect all error types for this claim
                error_types = [t["type"] for t in transform["applied_transformations"]]
                error_types_str = ",".join(error_types)
                
                rows.append({
                    "premise": original_claim,
                    "hypothesis": transformed_claim,
                    "error_types": error_types_str,
                    "label": "contradiction"
                })
    
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
    return len(rows)


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
    
    if INPUT_MODE == "file":
        # Single file mode
        output_csv = os.path.join(OUTPUT_FOLDER, "parsed_data.csv")
        parse_json_to_csv(INPUT_PATH, output_csv)
        
    elif INPUT_MODE == "directory":
        # Directory mode - process train/val/test files
        input_dir = Path(INPUT_PATH)
        total_samples = 0
        
        for json_file in input_dir.glob("*.json"):
            filename = json_file.stem.lower()
            
            # Determine output name based on train/val/test in filename
            if "train" in filename:
                output_name = "train.csv"
            elif "val" in filename or "validation" in filename:
                output_name = "val.csv"
            elif "test" in filename:
                output_name = "test.csv"
            else:
                output_name = f"{filename}.csv"
            
            output_csv = os.path.join(OUTPUT_FOLDER, output_name)
            count = parse_json_to_csv(str(json_file), output_csv)
            total_samples += count
        
        print(f"\n🎉 Total samples processed: {total_samples}")
    
    else:
        raise ValueError(f"Invalid INPUT_MODE: {INPUT_MODE}. Use 'file' or 'directory'")


if __name__ == "__main__":
    print("="*50)
    print("AUGMENTATION JSON TO CSV PARSER")
    print("="*50)
    print(f"Input mode: {INPUT_MODE}")
    print(f"Input path: {INPUT_PATH}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("="*50 + "\n")
    
    main()