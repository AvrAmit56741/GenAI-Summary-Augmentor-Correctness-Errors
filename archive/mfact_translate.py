from openai import OpenAI
from typing import List, Dict
import pandas as pd
import time
from tqdm import tqdm
import html
import codecs
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue


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


def translate_row(row_data: dict, api_key: str) -> dict:
    """
    Translates a single row (premise and hypothesis) to Hebrew.

    Args:
        row_data: Dictionary containing row index, premise and hypothesis
        api_key: OpenAI API key

    Returns:
        Dictionary with translated content and original index
    """
    client = OpenAI(api_key=api_key)

    try:
        # Translate premise
        premise_response = client.chat.completions.create(
            model="gpt-4o-mini",
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

        # Translate hypothesis
        hypothesis_response = client.chat.completions.create(
            model="gpt-4o-mini",
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
    output_file = "/home/avramit/mfact-summ/mfact/datasets/heb/he-translated-train.csv"
    num_rows = 10000
    max_workers = 10  # Limit concurrent API calls to avoid rate limits

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
        print(f"\nTranslating {num_rows} rows using {max_workers} concurrent threads...")

        # Prepare data for threading
        translation_tasks = []
        for idx in range(num_rows):
            row_data = {
                'index': idx,
                'premise': df.iloc[idx]['premise'],
                'hypothesis': df.iloc[idx]['hypothesis'],
                'label': df.iloc[idx]['label']
            }
            translation_tasks.append(row_data)

        # Store results in order
        results = [None] * num_rows

        # Create thread pool and execute translations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(translate_row, task, api_key): task['index']
                for task in translation_tasks
            }

            # Process completed translations with progress bar
            for future in tqdm(as_completed(future_to_idx), total=num_rows, desc="Translating rows"):
                result = future.result()
                results[result['index']] = result

        # Create final dataframe from results
        translated_df = pd.DataFrame(results)

        # Save the translated data
        translated_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nTranslation completed! Translated file saved to: {output_file}")

        # Print summary
        print(f"\nSummary:")
        print(f"Total rows processed: {num_rows}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()