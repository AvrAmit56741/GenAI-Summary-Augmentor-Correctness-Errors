from openai import OpenAI
from openai import AzureOpenAI
from typing import List, Dict
import pandas as pd
import time
from tqdm import tqdm
import html
import codecs
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from summary_augmentation.api_keys import AZURE_OPENAI_API_KEY_4o  # loaded from .env
AZURE_OPENAI_ENDPOINT_4o=r"https://lior-openai2.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01-preview"

ak = AZURE_OPENAI_API_KEY_4o
ep = AZURE_OPENAI_ENDPOINT_4o

client = AzureOpenAI(azure_endpoint=ep, api_key=ak, api_version="2024-02-01")



def clean_translation(translated_text: str) -> str:
    """Remove any potential prompt text from the translation."""
    # Remove common prompt markers and their content
    markers = [
        "[System Instructions]",
        "[Translation Target Style]",
        "[Translation Task]",
        "[Output Format]",
        "Translate this text to Hebrew:",
        "Respond with only",
        "Here's the translation of the provided text into Hebrew:"
    ]

    cleaned_text = translated_text
    for marker in markers:
        cleaned_text = cleaned_text.replace(marker, "")

    # Remove any leading/trailing whitespace
    return cleaned_text.strip()

def decode_hebrew_text(text):
    """
    Decode garbled Hebrew text to proper Hebrew characters.
    """
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        try:
            return text.encode('cp1252').decode('utf-8')
        except:
            return text


def translate_row(row_data: dict) -> dict:
    """
    Translates a single row (premise and hypothesis) to Hebrew.

    Args:
        row_data: Dictionary containing row index, premise and hypothesis
        api_key: OpenAI API key

    Returns:
        Dictionary with translated content and original index
    """
    client = AzureOpenAI(azure_endpoint=ep, api_key=ak, api_version="2024-02-01")

    try:
        # Translate premise
        premise_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        """
                        "You are a professional translator specializing in English-to-Hebrew translation, "
                        "with expertise in named entity recognition and preservation. "
                        "Your task is to produce an accurate, natural, and fluent Hebrew translation of the user-provided text, "
                        "while carefully retaining the original meaning, tone, and style. "
                        "Do not alter or mistranslate the names of people, organizations, companies, or legal terms. "
                        "You may adjust sentence structure for readability, but do not omit any crucial details or add new information."
                        "Ensure the translation maintains the Original meaning and intent, Professional / formal tone where appropriate, Natural Hebrew sentence structure,Proper Hebrew grammar and syntax"
                        """
                    )
                },
                {
                    "role": "user",
                    "content": f"Please translate the following English text into Hebrew:\n\nText to translate:\n{row_data['premise']}"
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        time.sleep(0.3)
        # Translate hypothesis
        hypothesis_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        """
                        "You are a professional translator specializing in English-to-Hebrew translation, "
                        "with expertise in named entity recognition and preservation. "
                        "Your task is to produce an accurate, natural, and fluent Hebrew translation of the user-provided text, "
                        "while carefully retaining the original meaning, tone, and style. "
                        "Do not alter or mistranslate the names of people, organizations, companies, or legal terms. "
                        "You may adjust sentence structure for readability, but do not omit any crucial details or add new information."
                        "Ensure the translation maintains the Original meaning and intent, Professional / formal tone where appropriate, Natural Hebrew sentence structure,Proper Hebrew grammar and syntax"
                        """
                    )
                },
                {
                    "role": "user",
                    "content": f"Please translate the following English text into Hebrew:\n\nText to translate:\n{row_data['hypothesis']}"
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )

        return {
            'index': row_data['index'],
            'premise': clean_translation(premise_response.choices[0].message.content),
            'hypothesis': clean_translation(hypothesis_response.choices[0].message.content),
            'label': row_data['label']
        }

    except Exception as e:
        print(f"Error translating row {row_data['index']}: {str(e)}")
        return {
            'index': row_data['index'],
            'premise': row_data['premise'],
            'hypothesis': row_data['hypothesis'],
            'label': row_data['label']
        }


def main():
    # Hardcoded configuration
    #api_key = "<redacted - see .env>"
    input_file = "/home/avramit/mfact-summ/mfact/datasets/en-translated-train.csv"
    output_file = "/home/avramit/mfact-summ/mfact/datasets/heb/he-translated-train-34000-40000.csv"
    num_rows = 6000
    max_workers = 5 # Limit concurrent API calls to avoid rate limits
    start_row = 34000
    save_interval = 500  # Save every 500 translations
    # Get user input for starting row
    # try:
    #     user_start = input("Enter starting row number (default: 0): ")
    #     if user_start.strip():
    #         start_row = int(user_start)
    #         print(f"Starting translation from row {start_row}")
    # except ValueError:
    #     print(f"Invalid input. Starting from row 0.")
    #     start_row = 0

    try:
        # Read and decode the CSV file
        encodings = ['utf-8', 'utf-8-sig', 'cp1255', 'latin1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(input_file, encoding=encoding)
                test_text = df['premise'].iloc[0]
                decoded_text = decode_hebrew_text(test_text)
                if any('\u0590' <= c <= '\u05FF' for c in decoded_text):
                    print(f"Successfully read file with encoding: {encoding}")
                    break
            except:
                continue

        if df is None:
            raise Exception("Could not read the CSV file with any encoding")

        # Decode Hebrew text
        df['premise'] = df['premise'].apply(decode_hebrew_text)
        df['hypothesis'] = df['hypothesis'].apply(decode_hebrew_text)

        # Validate number of rows
        num_rows = min(num_rows, len(df))
        print(
            f"\nTranslating {num_rows} rows (from row {start_row} to {start_row + num_rows - 1}) using {max_workers} concurrent threads...")

        # Prepare data for threading
        translation_tasks = []
        # Calculate the end row based on start_row and num_rows
        end_row = min(start_row + num_rows, len(df))
        # Adjust num_rows to actual count
        num_rows = end_row - start_row

        for idx in range(start_row, end_row):
            row_data = {
                'index': idx - start_row,  # Relative index for results array
                'original_index': idx,  # Original index in the dataframe
                'premise': df.iloc[idx]['premise'],
                'hypothesis': df.iloc[idx]['hypothesis'],
                'label': df.iloc[idx]['label']
            }
            translation_tasks.append(row_data)

        # Store results in order
        results = [None] * num_rows
        completed_count = 0

        # Create thread pool and execute translations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(translate_row, task): task['index']
                for task in translation_tasks
            }

            # Process completed translations with progress bar
            for future in tqdm(as_completed(future_to_idx), total=num_rows, desc="Translating rows"):
                result = future.result()
                results[result['index']] = result

                # Increment completed count
                completed_count += 1

                # Save intermediate results every save_interval translations
                if completed_count % save_interval == 0:
                    # Create temporary dataframe from current valid results
                    valid_results = [r for r in results if r is not None]
                    temp_df = pd.DataFrame(valid_results)

                    # Save to CSV
                    temp_df.to_csv(output_file, index=False, encoding='utf-8')
                    print(f"\nIntermediate save: {completed_count}/{num_rows} translations saved to {output_file}")

        # Create final dataframe from all results and save
        translated_df = pd.DataFrame(results)
        translated_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nTranslation completed! Final translated file saved to: {output_file}")

        # Print summary
        print(f"\nSummary:")
        print(f"Total rows processed: {num_rows}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
#     try:
#         # Read and decode the CSV file
#         encodings = ['utf-8', 'utf-8-sig', 'cp1255', 'latin1']
#         df = None
#
#         for encoding in encodings:
#             try:
#                 df = pd.read_csv(input_file, encoding=encoding)
#                 test_text = df['premise'].iloc[0]
#                 decoded_text = decode_hebrew_text(test_text)
#                 if any('\u0590' <= c <= '\u05FF' for c in decoded_text):
#                     print(f"Successfully read file with encoding: {encoding}")
#                     break
#             except:
#                 continue
#
#         if df is None:
#             raise Exception("Could not read the CSV file with any encoding")
#
#         # Decode Hebrew text
#         df['premise'] = df['premise'].apply(decode_hebrew_text)
#         df['hypothesis'] = df['hypothesis'].apply(decode_hebrew_text)
#
#         # Validate number of rows
#         num_rows = min(num_rows, len(df))
#         print(f"\nTranslating {num_rows} rows using {max_workers} concurrent threads...")
#
#         # Prepare data for threading
#         translation_tasks = []
#         for idx in range(num_rows):
#             row_data = {
#                 'index': idx,
#                 'premise': df.iloc[idx]['premise'],
#                 'hypothesis': df.iloc[idx]['hypothesis'],
#                 'label': df.iloc[idx]['label']
#             }
#             translation_tasks.append(row_data)
#
#         # Store results in order
#         results = [None] * num_rows
#
#         # Create thread pool and execute translations
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Submit all tasks
#             future_to_idx = {
#                 executor.submit(translate_row, task): task['index']
#                 for task in translation_tasks
#             }
#
#             # Process completed translations with progress bar
#             for future in tqdm(as_completed(future_to_idx), total=num_rows, desc="Translating rows"):
#                 result = future.result()
#                 results[result['index']] = result
#
#         # Create final dataframe from results
#         translated_df = pd.DataFrame(results)
#
#         # Save the translated data
#         translated_df.to_csv(output_file, index=False, encoding='utf-8')
#         print(f"\nTranslation completed! Translated file saved to: {output_file}")
#
#         # Print summary
#         print(f"\nSummary:")
#         print(f"Total rows processed: {num_rows}")
#
#     except Exception as e:
#         print(f"Error: {str(e)}")
#
#
