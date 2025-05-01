
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import logging

from semantic_router.encoders import HuggingFaceEncoder
from semantic_chunkers import StatisticalChunker

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = SentenceTransformer(embedding_model_name)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Ensure this is set to INFO or DEBUG to capture token counts
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
# Define model and tokenizer
model_name = "dicta-il/dictalm2.0-instruct"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
print("Model loaded successfully.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully.")
# Initialize the encoder
encoder_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
logging.info(f"Initializing HuggingFaceEncoder with model '{encoder_model_name}'...")
encoder = HuggingFaceEncoder(model_name=encoder_model_name)
logging.info("HuggingFaceEncoder initialized successfully.")

# Initialize the StatisticalChunker with the encoder
#chunker = StatisticalChunker(encoder=encoder)  # CHANGE: Initialize StatisticalChunker instead of ConsecutiveChunker
logging.info("StatisticalChunker initialized successfully.")
chunker = StatisticalChunker(
    encoder=encoder,
    min_split_tokens=250,
    max_split_tokens=500,  # Allow more deviation from the threshold
)


def remove_overlap_simple(translated_chunks):
    """
    Removes the first two sentences from each translated chunk (except the first one).

    Args:
        translated_chunks (list of str): List of translated chunks.

    Returns:
        list of str: The translated chunks with overlaps removed.
    """
    cleaned_chunks = []

    for idx, chunk in enumerate(translated_chunks):
        if idx == 0:
            # Keep the first chunk unchanged
            cleaned_chunks.append(chunk)
        else:
            # Remove the first two sentences from subsequent chunks
            sentences = re.split(r'(?<=[.!?]) +', chunk)
            cleaned_chunk = ' '.join(sentences[2:])  # Skip the first 2 sentences
            cleaned_chunks.append(cleaned_chunk)

    return cleaned_chunks


def remove_overlapping_text(prev_translated, current_translated, max_overlap_sentences=3, similarity_threshold=0.78):
    """
    Removes overlapping text from the current translated chunk by comparing the first 4 sentences of the current chunk
    with the last 4 sentences of the previous chunk.

    Args:
        prev_translated (str): The translated text of the previous chunk.
        current_translated (str): The translated text of the current chunk.
        max_overlap_sentences (int): Number of sentences to compare for overlap.
        similarity_threshold (float): Cosine similarity threshold to consider sentences as overlapping.

    Returns:
        str: The current translated chunk with overlapping sentences removed.
    """
    logging.info("Starting overlap detection between consecutive chunks.")
    logging.info(f"Previous chunk: {prev_translated}")
    logging.info(f"Current chunk: {current_translated}")

    # Split the translated texts into sentences
    prev_sentences = re.split(r'(?<=[.!?]) +', prev_translated)
    current_sentences = re.split(r'(?<=[.!?]) +', current_translated)

    logging.info(f"Previous chunk split into {len(prev_sentences)} sentences.")
    logging.info(f"Current chunk split into {len(current_sentences)} sentences.")

    # Limit to the last 4 sentences of the previous chunk and the first 4 sentences of the current chunk
    prev_overlap_sentences = prev_sentences[-max_overlap_sentences:]
    current_overlap_sentences = current_sentences[:max_overlap_sentences]

    logging.info(f"Comparing the first {len(current_overlap_sentences)} sentences of the current chunk "
                 f"with the last {len(prev_overlap_sentences)} sentences of the previous chunk.")

    # Iterate over each sentence in the current chunk's first 4 sentences
    for curr_idx, current_sentence in enumerate(current_overlap_sentences):
        logging.info(f"Checking current sentence {curr_idx + 1}: '{current_sentence}'")
        for prev_idx, prev_sentence in enumerate(prev_overlap_sentences):
            logging.info(f"Comparing with previous sentence {prev_idx + 1}: '{prev_sentence}'")

            # Compute embeddings
            prev_embedding = embedding_model.encode(prev_sentence, convert_to_tensor=True)
            current_embedding = embedding_model.encode(current_sentence, convert_to_tensor=True)

            # Compute cosine similarity
            similarity = cosine_similarity(
                prev_embedding.cpu().numpy().reshape(1, -1),
                current_embedding.cpu().numpy().reshape(1, -1)
            )[0][0]

            logging.info(f"Similarity between current sentence {curr_idx + 1} and previous sentence {prev_idx + 1}: {similarity:.4f}")

            # Remove the sentence if the similarity exceeds the threshold
            if similarity >= similarity_threshold:
                logging.info(f"Removing sentence from current chunk: '{current_sentence}' (similarity: {similarity:.4f})")
                current_sentences.remove(current_sentence)
                break
        else:
            logging.info(f"No significant overlap found for current sentence: '{current_sentence}'")

    # Log the final cleaned chunk
    cleaned_chunk = ' '.join(current_sentences)
    logging.info(f"Final cleaned current chunk: {cleaned_chunk}")

    # Return the cleaned current chunk
    return cleaned_chunk


# def remove_overlapping_text(prev_translated, current_translated, max_overlap_sentences=3, similarity_threshold=0.75):
#     """
#     Dynamically removes overlapping text from the current translated chunk based on similarity with the previous translated chunk.
#
#     Args:
#         prev_translated (str): The translated text of the previous chunk.
#         current_translated (str): The translated text of the current chunk.
#         max_overlap_sentences (int): Maximum number of sentences to consider for overlap.
#         similarity_threshold (float): Cosine similarity threshold to consider as overlap.
#
#     Returns:
#         str: The current translated chunk with overlapping text removed if similarity exceeds the threshold.
#     """
#     # Split the translated texts into sentences
#     prev_sentences = re.split(r'(?<=[.!?]) +', prev_translated)
#     current_sentences = re.split(r'(?<=[.!?]) +', current_translated)
#
#     # Iterate from max_overlap_sentences down to 1
#     for overlap_size in range(max_overlap_sentences, 0, -1):
#         if len(prev_sentences) < overlap_size or len(current_sentences) < overlap_size:
#             continue  # Not enough sentences to compare
#
#         # Get the last 'overlap_size' sentences from the previous translation
#         prev_overlap = ' '.join(prev_sentences[-overlap_size:])
#
#         # Get the first 'overlap_size' sentences from the current translation
#         current_overlap = ' '.join(current_sentences[:overlap_size])
#
#         # Compute embeddings
#         prev_embedding = embedding_model.encode(prev_overlap, convert_to_tensor=True)
#         current_embedding = embedding_model.encode(current_overlap, convert_to_tensor=True)
#
#         # Compute cosine similarity
#         similarity = cosine_similarity(prev_embedding.cpu().numpy().reshape(1, -1),
#                                        current_embedding.cpu().numpy().reshape(1, -1))[0][0]
#
#         logging.info(f"Overlap size {overlap_size}: Similarity = {similarity:.4f}")
#
#         if similarity >= similarity_threshold:
#             # Print the overlapping sentences that are identified for removal
#             print(f"Sentence(s) in previous chunk ({overlap_size} sentence{'s' if overlap_size > 1 else ''}): {prev_overlap}")
#             print(f"Sentence(s) in current chunk to remove ({overlap_size} sentence{'s' if overlap_size > 1 else ''}): {current_overlap}")
#
#             # Remove the overlapping sentences from the current translation
#             current_sentences = current_sentences[overlap_size:]
#             logging.info(f"Removed {overlap_size} overlapping sentence{'s' if overlap_size > 1 else ''} from current chunk.")
#
#             return ' '.join(current_sentences)
#
#     # If no significant overlap detected, keep all sentences
#     logging.info("No significant overlap detected. Keeping all sentences.")
#     logging.info(f"prev chunk x {prev_translated}")
#     logging.info(f"current chunk x+1 {current_translated}")
#     return current_translated

def chunk_and_translate(text, line):
    """
    Performs semantic chunking using StatisticalChunker.
    If a chunk contains more than 7 sentences, it splits it further using overlapping chunks.
    Logs the number of tokens and translates each chunk.

    Args:
        text (str): The original Hebrew text to be chunked and translated.

    Returns:
        str: The concatenated translated text.
    """
    logging.info("Performing semantic chunking using StatisticalChunker.")

    try:
        # Perform semantic chunking; the chunker expects a list of documents
        chunks = chunker(docs=[text])
        semantic_chunks = chunks[0]  # Extract the list of chunks for the first (and only) document
        chunk_count = len(semantic_chunks)
        logging.warning(f"line {line} split to {chunk_count} chunks")

        # If more than 13 chunks, rechunk with a stricter configuration
        if chunk_count < 3:
            logging.warning(f"Line {line} has {chunk_count} chunks. loss rechunking with loss parameters.")
            loss_chunker = StatisticalChunker(encoder=encoder, min_split_tokens=70, max_split_tokens=200)
            chunks = loss_chunker(docs=[text])
            semantic_chunks = chunks[0]
            chunk_count = len(semantic_chunks)
            logging.warning(f"After rechunking, line {line} split into {chunk_count} chunks.")

        elif chunk_count > 13:
            logging.warning(f"Line {line} has {chunk_count} chunks. stricter rechunking with stricter parameters.")
            stricter_chunker = StatisticalChunker(encoder=encoder, min_split_tokens=500, max_split_tokens=700)
            chunks = stricter_chunker(docs=[text])
            semantic_chunks = chunks[0]
            chunk_count = len(semantic_chunks)
            logging.warning(f"After rechunking, line {line} split into {chunk_count} chunks.")

        # Apply overlap between consecutive chunks
        for i in range(len(semantic_chunks) - 1):
            last_sentences = re.split(r'(?<=[.!?]) +', semantic_chunks[i].splits[-1])[-2:]
            semantic_chunks[i + 1].splits = last_sentences + semantic_chunks[i + 1].splits

        write_chunks_to_files(semantic_chunks, line)
        translated_chunks = []

        # Start translating each chunk
        for idx, chunk in enumerate(semantic_chunks, start=1):
            chunk_text = " ".join(chunk.splits)
            if not chunk_text:
                logging.warning(f"Chunk {idx} is empty after splitting. Skipping translation.")
                continue

            sentences = re.split(r'(?<=[.!?]) +', chunk_text)
            sentence_count = len(sentences)

            if sentence_count >= 13:
                logging.info(f"Chunk {idx} has {sentence_count} sentences. Splitting with overlap. line: {line}")
                translated_chunk = chunk_and_translate_with_overlap(chunk_text, max_sentences=13, overlap_sentences=3)
            elif (13 >= sentence_count >= 7) and chunk.token_count > 3000:
                logging.info(f"Chunk {idx} has handle long sentences: {sentence_count} sentences. Splitting with overlap. line: {line}")
                translated_chunk = chunk_and_translate_with_overlap(chunk_text, max_sentences=7, overlap_sentences=4)
            else:
                token_count = len(tokenizer.tokenize(chunk_text))
                logging.info(f"Chunk {idx} of line: {line}:, consists of {token_count} tokens.")
                translated_chunk = translate_text(chunk_text)

            translated_chunks.append(translated_chunk)
            logging.debug(f"Translated Chunk {idx}: {translated_chunk}. line: {line}")

        # Remove overlap from translated chunks before concatenating
        for i in range(1, len(translated_chunks)):
            translated_chunks[i] = remove_overlapping_text(translated_chunks[i - 1], translated_chunks[i])

        translated_text = ' '.join(translated_chunks)
        logging.info(f"All chunks of line: {line} translated successfully.")

        return translated_text

    except Exception as e:
        logging.error(f"Error during semantic chunking and translation: {e}")
        return "Translation Error"


def preprocess_text(text):
    """
    Preprocesses the input text by handling bullets, special characters,
    measurements, and formatting to enhance translation quality.
    """
    # 1. Replace bullet points with newline and dash
    text = re.sub(r'•\s*', '- ', text)

    # 2. Normalize measurements (e.g., "50 גרם חמאה קרה" -> "50 grams of cold butter")
    # This step can be customized based on specific needs
    # For simplicity, we'll leave measurements as-is

    # 3. Remove problematic symbols (e.g., excessive quotes, backslashes)
    text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\', '')

    # 4. Replace multiple spaces and newlines with single space or standardized delimiter
    text = re.sub(r'\s+', ' ', text)

    # 5. Handle quotes and parentheses
    text = re.sub(r'\"', '"', text)
    text = re.sub(r'\(', '(', text)
    text = re.sub(r'\)', ')', text)

    # 6. Optionally, separate ingredients and instructions with clear delimiters
    # For example, replace "לתחתית העוגה" with "\nIngredients:\n"
    # This requires knowledge of the text structure

    # 7. Trim leading and trailing whitespace
    text = text.strip()

    return text


def translate_text(text):
    print(f"Translating text: {text}")

    preprocessed_text = preprocess_text(text)

    print(f"Translating preprocessed text: {preprocessed_text}")  # Log a snippet

    prompt = f"<s>[INST] You are a translator from Hebrew to English. Translate the following text in Hebrew to English: '{preprocessed_text}' [/INST]"

    # Ensure the pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32768  # Match the model's max_position_embeddings
    )
    input_ids = encoded.input_ids.cuda()
    attention_mask = encoded.attention_mask.cuda()

    try:
        # Generate translation with a limit on new tokens
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=4000,  # Adjust based on expected output size
            early_stopping=True
        )

        # Decode and clean up the output
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Raw translated text: {translated_text}")

        # Remove the prompt text from the beginning, if present
        if "[INST]" in translated_text and "[/INST]" in translated_text:
            translated_text = translated_text.split("[/INST]")[1].strip()

        print(f"Cleaned translated text: {translated_text}")

        # Handle any remaining Hebrew sentences
        final_translation = handle_remaining_hebrew(translated_text)

        return final_translation

    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation Error"

import os
def write_chunks_to_files(semantic_chunks, line_number, output_dir="."):
    """
    Writes all chunks of a single line to a single JSON file in the specified output directory.

    Args:
        semantic_chunks (list): List of chunks generated from the chunking process.
        line_number (int): The line number of the JSONL being processed.
        output_dir (str): Directory to save the chunk files. Defaults to the current directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the list of chunks to be written to the file
    chunks_data = []
    for idx, chunk in enumerate(semantic_chunks, start=1):
        chunk_text = " ".join(chunk.splits).strip()
        if not chunk_text:
            logging.warning(f"Chunk {idx} of line {line_number} is empty. Skipping.")
            continue

        chunks_data.append({
            "chunk_index": idx,
            "text": chunk_text
        })

    # Create a filename for the chunks file in the specified output directory
    chunk_filename = os.path.join(output_dir, f"line_{line_number}.json")

    # Write all chunks to a single file
    with open(chunk_filename, 'w', encoding='utf-8') as chunk_file:
        json.dump({"line": line_number, "chunks": chunks_data}, chunk_file, ensure_ascii=False, indent=4)

    logging.info(f"All chunks of line {line_number} written to {chunk_filename}.")


def handle_remaining_hebrew(translated_text):
    max_retries = 3  # Maximum number of retry attempts
    attempt = 0

    while attempt < max_retries:
        # Identify sentences with Hebrew characters
        sentences = re.split(r'(?<=[.!?]) +', translated_text)
        hebrew_sentences = [s for s in sentences if re.search(r'[א-ת]', s)]

        if not hebrew_sentences:
            break  # No Hebrew sentences left

        print(f"Attempt {attempt + 1}: Translating {len(hebrew_sentences)} Hebrew sentence(s).")

        for hebrew_sentence in hebrew_sentences:
            print(f"Translating sentence: {hebrew_sentence}")
            # Reuse the same prompt format
            translated_sentence = translate_sentence(hebrew_sentence)
            print(f"Translated sentence: {translated_sentence}")

            # Replace the Hebrew sentence with its translation
            translated_text = translated_text.replace(hebrew_sentence, translated_sentence)

        attempt += 1

    # Final check after retries
    if re.search(r'[א-ת]', translated_text):
        print("Error: Translation still contains Hebrew characters after maximum retries.")
        translated_text = re.sub(r'[א-ת]', '', translated_text)
        print(f"Final post-processed translated text: {translated_text}")

    return translated_text


def translate_sentence(sentence):
    prompt = f"<s>[INST] You are a translator from Hebrew to English. Translate the following text in Hebrew to English: '{sentence}' [/INST]"

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = encoded.input_ids.cuda()
    attention_mask = encoded.attention_mask.cuda()

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=300,
        early_stopping=True
    )

    translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[INST]" in translated_sentence and "[/INST]" in translated_sentence:
        translated_sentence = translated_sentence.split("[/INST]")[1].strip()

    return translated_sentence


# def chunk_and_translate(text, line):
#     """
#     Performs semantic chunking using StatisticalChunker.
#     If a chunk contains more than 7 sentences, it splits it further using overlapping chunks.
#     Logs the number of tokens and translates each chunk.
#
#     Args:
#         text (str): The original Hebrew text to be chunked and translated.
#
#     Returns:
#         str: The concatenated translated text.
#     """
#     logging.info("Performing semantic chunking using StatisticalChunker.")
#
#     try:
#         # Perform semantic chunking; the chunker expects a list of documents
#         chunks = chunker(docs=[text])
#         semantic_chunks = chunks[0]  # Extract the list of chunks for the first (and only) document
#         chunk_count = len(semantic_chunks)
#         logging.warning(f"line {line} split to {chunk_count} chunks")
#         # If more than 13 chunks, rechunk with a stricter configuration
#         if chunk_count < 3:
#             logging.warning(f"Line {line} has {chunk_count} chunks. loss rechunking with loss parameters.")
#
#             # Rechunk with larger min_split_tokens and max_split_tokens
#             loss_chunker = StatisticalChunker(
#                 encoder=encoder,
#                 min_split_tokens=70,
#                 max_split_tokens=200
#             )
#             chunks = loss_chunker(docs=[text])
#             semantic_chunks = chunks[0]
#             chunk_count = len(semantic_chunks)
#
#             logging.warning(f"After rechunking, line {line} split into {chunk_count} chunks.")
#
#         elif chunk_count > 13:
#             logging.warning(f"Line {line} has {chunk_count} chunks. stricter rechunking with stricter parameters.")
#
#             # Rechunk with larger min_split_tokens and max_split_tokens
#             stricter_chunker = StatisticalChunker(
#                 encoder=encoder,
#                 min_split_tokens=500,
#                 max_split_tokens=700
#             )
#             chunks = stricter_chunker(docs=[text])
#             semantic_chunks = chunks[0]
#             chunk_count = len(semantic_chunks)
#
#             logging.warning(f"After rechunking, line {line} split into {chunk_count} chunks.")
#
#         write_chunks_to_files(semantic_chunks, line)
#         translated_chunks = []
#
#         for idx, chunk in enumerate(semantic_chunks, start=1):
#             # Extract text from the Chunk object
#             chunk_text = " ".join(chunk.splits)
#
#             if not chunk_text:
#                 logging.warning(f"Chunk {idx} is empty after splitting. Skipping translation.")
#                 continue  # Skip empty chunks
#
#             # Split the chunk into sentences
#             sentences = re.split(r'(?<=[.!?]) +', chunk_text)
#             sentence_count = len(sentences)
#
#             if sentence_count >= 13:
#                 logging.info(f"Chunk {idx} has {sentence_count} sentences. Splitting with overlap. line: {line}")
#                 # Use overlapping chunking based on sentences
#                 translated_chunk = chunk_and_translate_with_overlap(chunk_text, max_sentences=13, overlap_sentences=3)
#             # handle long sentences
#             elif (13 >= sentence_count >= 7) and chunk.token_count > 3000:
#                 logging.info(f"Chunk {idx} has handle long sentences: {sentence_count} sentences. Splitting with overlap. line: {line}")
#                 # Use overlapping chunking based on sentences
#                 translated_chunk = chunk_and_translate_with_overlap(chunk_text, max_sentences=7, overlap_sentences=4)
#             else:
#                 # Calculate the number of tokens in the chunk
#                 token_count = len(tokenizer.tokenize(chunk_text))
#
#                 # Log the token count
#                 logging.info(f"Chunk {idx} of line: {line}:, consists of {token_count} tokens.")
#
#                 # Translate the chunk
#                 translated_chunk = translate_text(chunk_text)
#
#             translated_chunks.append(translated_chunk)
#
#             # Optional: Log the translated chunk (can be commented out if verbose)
#             logging.debug(f"Translated Chunk {idx}: {translated_chunk}. line: {line}")
#
#         # Concatenate all translated chunks into a single string
#         translated_text = ' '.join(translated_chunks)
#         logging.info(f"All chunks of line: {line} translated successfully.")
#
#         return translated_text
#
#     except Exception as e:
#         logging.error(f"Error during semantic chunking and translation: {e}")
#         return "Translation Error"


def chunk_and_translate_with_overlap(text, max_sentences=7, overlap_sentences=2):
    """
    Splits text into overlapping chunks based on the number of sentences.
    Each chunk contains up to `max_sentences` with an overlap of `overlap_sentences`.
    Removes the first `overlap_sentences` from each translated chunk to avoid duplication.

    Args:
        text (str): The original Hebrew text to be split and translated.
        max_sentences (int): Maximum number of sentences per chunk.
        overlap_sentences (int): Number of sentences to overlap between chunks.

    Returns:
        str: The concatenated translated text without duplicated overlaps.
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    translated_chunks = []
    start = 0
    first_chunk = True

    while start < len(sentences):
        end = start + max_sentences
        chunk_sentences = sentences[start:end]
        chunk_text = ' '.join(chunk_sentences)

        # Translate the current chunk
        translated_chunk = translate_text(chunk_text)

        if not first_chunk:
            # Split the translated chunk into sentences
            translated_sentences = re.split(r'(?<=[.!?]) +', translated_chunk)
            if len(translated_sentences) > overlap_sentences:
                # Remove the first `overlap_sentences` sentences
                translated_chunk = ' '.join(translated_sentences[overlap_sentences:])
            else:
                # If translated chunk has <= overlap_sentences, skip adding to avoid duplicates
                translated_chunk = ''

        if translated_chunk:
            translated_chunks.append(translated_chunk)

        # Update for next chunk with overlap
        start += max_sentences - overlap_sentences
        first_chunk = False

    # Concatenate all translated chunks
    return ' '.join(translated_chunks)


# Function to process the JSONL file
def process_jsonl(input_path, output_path, use_overlap=False):
    print(f"Processing input file: {input_path}")
    allowed_doc_ids = [
        "food-6163018"
    ]
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        line_count = 0
        processed_count = 0

        for line in infile:
            line_count += 1

            # Skip lines outside the range 100 to 200
            if line_count < 200:
                continue
            if line_count > 220:
                break

            print(f"Processing line {line_count}: {line.strip()}")
            data = json.loads(line)

            # Translate the required fields if they exist
            if "text_raw" in data:
                data["text_raw_translated"] = chunk_and_translate(data["text_raw"], line_count)  # CHANGE: Use updated function

            if "summary" in data:
                print("Found 'summary' field.")
                data["summary_translated"] = translate_text(data["summary"])

            if "metadata" in data and "ai_summary" in data["metadata"]:
                print("Found 'metadata.ai_summary' field.")
                data["metadata"]["ai_summary_translated"] = translate_text(data["metadata"]["ai_summary"])

            # Write the updated JSON object to the output file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f"Processed line {line_count} successfully.")
            processed_count += 1

    print(f"Processing completed. Output saved to {output_path}")

# Paths to input and output files
input_jsonl_path = "/sise/home/avramit/pycharm_project_502/summarization-7-heb_up.jsonl"  # Replace with your input path
output_jsonl_path = "/sise/home/avramit/pycharm_project_502/output_new_2.jsonl"  # Replace with your output path

use_overlap = False  # Set to True if you plan to use the overlap function
# Process the input file
process_jsonl(input_jsonl_path, output_jsonl_path)
print("File processing completed!")



# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import re
#
# # Define model and tokenizer
# model_name = "dicta-il/dictalm2.0-instruct"
# print("Loading model and tokenizer...")
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
# print("Model loaded successfully.")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# print("Tokenizer loaded successfully.")
#
#
# def preprocess_text(text):
#     """
#     Preprocesses the input text by handling bullets, special characters,
#     measurements, and formatting to enhance translation quality.
#     """
#     # 1. Replace bullet points with newline and dash
#     text = re.sub(r'•\s*', '- ', text)
#
#     # 2. Normalize measurements (e.g., "50 גרם חמאה קרה" -> "50 grams of cold butter")
#     # This step can be customized based on specific needs
#     # For simplicity, we'll leave measurements as-is
#
#     # 3. Remove problematic symbols (e.g., excessive quotes, backslashes)
#     text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\', '')
#
#     # 4. Replace multiple spaces and newlines with single space or standardized delimiter
#     text = re.sub(r'\s+', ' ', text)
#
#     # 5. Handle quotes and parentheses
#     text = re.sub(r'\"', '"', text)
#     text = re.sub(r'\(', '(', text)
#     text = re.sub(r'\)', ')', text)
#
#     # 6. Optionally, separate ingredients and instructions with clear delimiters
#     # For example, replace "לתחתית העוגה" with "\nIngredients:\n"
#     # This requires knowledge of the text structure
#
#     # 7. Trim leading and trailing whitespace
#     text = text.strip()
#
#     return text
#
#
# def translate_text(text):
#     print(f"Translating text: {text}")
#
#     preprocessed_text = preprocess_text(text)
#
#     print(f"Translating preprocessed text: {preprocessed_text}")  # Log a snippet
#
#     prompt = f"<s>[INST] You are a translator from Hebrew to English. Translate the following text in Hebrew to English: '{preprocessed_text}' [/INST]"
#
#     # Ensure the pad token is set
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     encoded = tokenizer(
#         prompt,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=32768  # Match the model's max_position_embeddings
#     )
#     input_ids = encoded.input_ids.cuda()
#     attention_mask = encoded.attention_mask.cuda()
#
#     print(f"Generated input IDs: {input_ids}")
#     print(f"Attention mask: {attention_mask}")
#
#     try:
#         # Generate translation with a limit on new tokens
#         outputs = model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             do_sample=False,
#             max_new_tokens=3000,  # Adjust based on expected output size
#             early_stopping=True
#         )
#         print(f"Generated output IDs: {outputs}")
#
#
#         # Decode and clean up the output
#         translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(f"Raw translated text: {translated_text}")
#
#         # Remove the prompt text from the beginning, if present
#         if "[INST]" in translated_text and "[/INST]" in translated_text:
#             translated_text = translated_text.split("[/INST]")[1].strip()
#
#         # Log the cleaned translated text
#         print(f"Cleaned translated text: {translated_text}")
#
#         # Post-process to handle any remaining Hebrew text
#         if re.search(r'[א-ת]', translated_text):
#             print("Warning: Translated text still contains Hebrew characters.")
#             # Optionally, you can replace or flag the Hebrew text
#             translated_text = re.sub(r'[א-ת]', '', translated_text)
#             print(f"Post-processed translated text: {translated_text}")
#
#         return translated_text
#     except Exception as e:
#         print(f"Error during translation: {e}")
#         return "Translation Error"
#
# def chunk_and_translate(text):
#     # Split text into sentences
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     translated_chunks = []
#
#     # Process in chunks of 10 sentences
#     for i in range(0, len(sentences), 7):
#         chunk = ' '.join(sentences[i:i+7])
#         print(f"Translating chunk: {chunk}")
#         translated_chunk = translate_text(chunk)
#         print(f"Translated chunk: {translated_chunk}")
#         translated_chunks.append(translated_chunk)
#
#     # Concatenate all translated chunks
#     return ' '.join(translated_chunks)
#
# def chunk_and_translate_with_overlap(text, chunk_size=1000, overlap_size=200):
#     # Tokenize the text
#     tokens = tokenizer.tokenize(text)
#     chunks = []
#
#     # Create overlapping chunks
#     start_idx = 0
#     while start_idx < len(tokens):
#         end_idx = min(start_idx + chunk_size, len(tokens))
#         chunk = tokens[start_idx:end_idx]
#         chunks.append(chunk)
#         start_idx += chunk_size - overlap_size
#
#     translated_chunks = []
#     previous_context = ""  # To store the previous context for removing overlap
#     for chunk in chunks:
#         # Convert chunk back to text for translation
#         chunk_text = tokenizer.convert_tokens_to_string(chunk)
#         translated_chunk = translate_text(chunk_text)
#
#         # Remove the overlapping portion from the translated chunk
#         if previous_context and previous_context in translated_chunk:
#             translated_chunk = translated_chunk.replace(previous_context, "", 1).strip()
#
#         # Update the previous context
#         previous_context = translate_text(
#             tokenizer.convert_tokens_to_string(chunk[-overlap_size:])
#         )
#
#         translated_chunks.append(translated_chunk)
#
#     # Concatenate all translated chunks
#     return ' '.join(translated_chunks)
#
#
# # Function to process the JSONL file
# def process_jsonl(input_path, output_path, use_overlap=False):
#     print(f"Processing input file: {input_path}")
#     with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
#         line_count = 0
#         processed_count = 0
#         for line in infile:
#             line_count += 1
#             if processed_count >= 20:
#                 print("Processed 200 lines, stopping further processing.")
#                 break
#
#             print(f"Processing line {line_count}: {line.strip()}")
#             data = json.loads(line)
#
#             # Translate the required fields if they exist
#             if "text_raw" in data:
#                 if use_overlap:
#                     data["text_raw_translated"] = chunk_and_translate_with_overlap(data["text_raw"])
#                 else:
#                     data["text_raw_translated"] = chunk_and_translate(data["text_raw"])
#
#             if "summary" in data:
#                 print("Found 'summary' field.")
#                 data["summary_translated"] = translate_text(data["summary"])
#
#             if "metadata" in data and "ai_summary" in data["metadata"]:
#                 print("Found 'metadata.ai_summary' field.")
#                 data["metadata"]["ai_summary_translated"] = translate_text(data["metadata"]["ai_summary"])
#
#             # Write the updated JSON object to the output file
#             outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
#             print(f"Processed line {line_count} successfully.")
#             processed_count += 1
#
#     print(f"Processing completed. Output saved to {output_path}")
#
# # Paths to input and output files
# input_jsonl_path = "/sise/home/avramit/pycharm_project_502/summarization-7-heb_up.jsonl"  # Replace with your input path
# output_jsonl_path = "/sise/home/avramit/pycharm_project_502/output_new_2.jsonl"  # Replace with your output path
#
# use_overlap = True
# # Process the input file
# process_jsonl(input_jsonl_path, output_jsonl_path)
# print("File processing completed!")

# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import re
#
# # Define model and tokenizer
# model_name = "dicta-il/dictalm2.0-instruct"
# print("Loading model and tokenizer...")
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
# print("Model loaded successfully.")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# print("Tokenizer loaded successfully.")
#
# def translate_text(text):
#     print(f"Translating text: {text}")
#     max_context_size = 4096  # Example: adjust based on your model's actual limit
#     max_new_tokens = 1000    # Reserved for the generated tokens
#     max_input_size = max_context_size - max_new_tokens
#
#     prompt = f"<s>[INST] You are a professional translator from Hebrew to English. Translate the following Hebrew text accurately, preserving its meaning and avoiding creative additions: '{text}' [/INST]"
#
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     # Encode with truncation to fit within available input size
#     encoded = tokenizer(
#         prompt,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=max_input_size
#     )
#     input_ids = encoded.input_ids.cuda()
#     attention_mask = encoded.attention_mask.cuda()
#
#     try:
#         outputs = model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             do_sample=False,
#             max_new_tokens=max_new_tokens,
#             early_stopping=True
#         )
#         translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#         # Post-process the translation
#         if "[INST]" in translated_text and "[/INST]" in translated_text:
#             translated_text = translated_text.split("[/INST]")[1].strip()
#
#         # Remove any remaining Hebrew characters
#         if re.search(r'[א-ת]', translated_text):
#             print("Warning: Translated text still contains Hebrew characters.")
#             translated_text = re.sub(r'[א-ת]', '', translated_text).strip()
#
#         print(f"Final translated text: {translated_text}")
#         return translated_text
#     except Exception as e:
#         print(f"Error during translation: {e}")
#         return "Translation Error"
#
#
# def chunk_and_translate_with_sentence_overlap(text, sentences_per_chunk=7, overlap_sentences=2):
#     # Split text into sentences based on periods
#     sentences = re.split(r'(?<=\.)\s+', text)  # Split on period followed by space
#     sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings and whitespace
#
#     chunks = []
#     start_idx = 0
#
#     while start_idx < len(sentences):
#         # Create a chunk with overlap
#         end_idx = min(start_idx + sentences_per_chunk, len(sentences))
#         chunk = ' '.join(sentences[start_idx:end_idx])
#         print(f"Original chunk (sentences {start_idx} to {end_idx}): {chunk}")
#
#         # Translate the chunk
#         translated_chunk = translate_text(chunk)
#
#         # Remove the overlap context (first `overlap_sentences`)
#         overlap_removed_chunk = ' '.join(translated_chunk.split('. ')[overlap_sentences:]).strip()
#
#         # Log the overlap-removed chunk
#         print(f"Translated chunk (overlap removed): {overlap_removed_chunk}")
#
#         chunks.append({"original_chunk": chunk, "translated_chunk": overlap_removed_chunk})
#         start_idx += sentences_per_chunk - overlap_sentences
#
#     return chunks
#
# def chunk_and_translate_by_paragraphs(text):
#     # Split by paragraphs using newlines or special markers
#     paragraphs = re.split(r'\n{2,}|\•', text)  # Split on double newlines or bullet points
#     paragraphs = [p.strip() for p in paragraphs if p.strip()]  # Remove empty strings and strip whitespace
#     chunks = []
#
#     for paragraph in paragraphs:
#         print(f"Processing paragraph: {paragraph}")
#         translated_paragraph = translate_text(paragraph)  # Translate the paragraph
#         chunks.append({"original_chunk": paragraph, "translated_chunk": translated_paragraph})
#
#     return chunks
#
# def chunk_and_translate(text):
#     sentences = re.split(r'\.\s*', text)  # Splits on a period followed by optional spaces
#     sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings and strip whitespace
#
#     chunks = []
#     print(sentences)
#     print(len(sentences))
#
#     for i in range(0, len(sentences), 5):
#         chunk = ' '.join(sentences[i:i+5])
#         translated_chunk = translate_text(chunk)
#         chunks.append({"original_chunk": chunk, "translated_chunk": translated_chunk})
#         print(f"Original chunk: {chunk}")
#         print(f"Translated chunk: {translated_chunk}")
#
#     return chunks
#
# def chunk_and_translate_with_overlap(text, chunk_size=1000, overlap_size=200):
#     tokens = tokenizer.tokenize(text)
#     chunks = []
#     start_idx = 0
#     previous_translation = ""  # To keep track of the previous chunk translation
#
#     while start_idx < len(tokens):
#         end_idx = min(start_idx + chunk_size, len(tokens))
#         chunk = tokens[start_idx:end_idx]
#         chunk_text = tokenizer.convert_tokens_to_string(chunk)
#         print(f"Original chunk (tokens {start_idx} to {end_idx}): {chunk_text}")
#
#         # Translate the chunk
#         translated_chunk = translate_text(chunk_text)
#
#         # Remove overlap redundancy if detected
#         if previous_translation and translated_chunk.startswith(previous_translation):
#             print("Detected overlap in translation. Removing redundant text.")
#             translated_chunk = translated_chunk[len(previous_translation):].strip()
#
#         # Log the translated chunk
#         print(f"Translated chunk: {translated_chunk}")
#
#         # Append the processed chunk
#         chunks.append({"original_chunk": chunk_text, "translated_chunk": translated_chunk})
#         previous_translation = translated_chunk[-overlap_size:]  # Update with the last part of the current translation
#         start_idx += chunk_size - overlap_size
#
#     return chunks
#
#
#
# def process_jsonl(input_path, output_path, chunk_output_path, use_overlap=False, use_paragraphs=False, use_sentence_overlap=False):
#     print(f"Processing input file: {input_path}")
#     chunk_data = []
#     max_elements = 15  # Process only the first 2 elements for testing
#     processed_count = 0
#
#     with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
#         for line in infile:
#             if processed_count >= max_elements:
#                 break
#
#             data = json.loads(line)
#
#             if "text_raw" in data:
#                 if use_paragraphs:
#                     # Use the paragraph-based translation
#                     chunks = chunk_and_translate_by_paragraphs(data["text_raw"])
#                 elif use_sentence_overlap:
#                     # Use sentence-based chunking with overlap for translation
#                     chunks = chunk_and_translate_with_sentence_overlap(data["text_raw"])
#                 elif use_overlap:
#                     # Use token-based overlapping chunks for translation
#                     chunks = chunk_and_translate_with_overlap(data["text_raw"])
#                 else:
#                     # Use sentence-based chunking for translation
#                     chunks = chunk_and_translate(data["text_raw"])
#
#                 data["text_raw_translated"] = " ".join(chunk["translated_chunk"] for chunk in chunks)
#                 chunk_data.append({"document_id": data.get("id", None), "chunks": chunks})
#
#                 if "metadata" in data and "summary" in data:
#                     print("Found 'summary' field.")
#                     data["summary_translated"] = translate_text(data["summary"])
#
#                 if "metadata" in data and "ai_summary" in data["metadata"]:
#                     print("Found 'metadata.ai_summary' field.")
#                     data["metadata"]["ai_summary_translated"] = translate_text(data["metadata"]["ai_summary"])
#
#             outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
#             processed_count += 1
#
#     with open(chunk_output_path, 'w', encoding='utf-8') as chunk_file:
#         json.dump(chunk_data, chunk_file, ensure_ascii=False, indent=4)
#
#     print(f"Processing completed. Processed {processed_count} elements. Output saved to {output_path} and {chunk_output_path}")
#
#
#
# # Paths to input and output files
# input_jsonl_path = "/sise/home/avramit/pycharm_project_502/summarization-7-heb_up.jsonl"  # Replace with your input path
# output_jsonl_path = "/sise/home/avramit/pycharm_project_502/output.jsonl"  # Replace with your output path
# chunk_json_path = "/sise/home/avramit/pycharm_project_502/chunks.json"  # Replace with your chunk output path
#
# use_overlap = True
# # Process the input file
# process_jsonl(input_jsonl_path, output_jsonl_path, chunk_json_path, True)
#
# print("File processing completed!")

# def translate_text(text):
#     print(f"Translating text: {text}")
#
#     preprocessed_text = preprocess_text(text)
#
#     print(f"Translating preprocessed text: {preprocessed_text}")  # Log a snippet
#
#     prompt = f"<s>[INST] You are a translator from Hebrew to English. Translate the following text in Hebrew to English: '{preprocessed_text}' [/INST]"
#
#     # Ensure the pad token is set
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     encoded = tokenizer(
#         prompt,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=32768  # Match the model's max_position_embeddings
#     )
#     input_ids = encoded.input_ids.cuda()
#     attention_mask = encoded.attention_mask.cuda()
#
#     #print(f"Generated input IDs: {input_ids}")
#     #print(f"Attention mask: {attention_mask}")
#
#     try:
#         # Generate translation with a limit on new tokens
#         outputs = model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             do_sample=False,
#             max_new_tokens=4000,  # Adjust based on expected output size
#             early_stopping=True
#         )
#         #print(f"Generated output IDs: {outputs}")
#
#
#         # Decode and clean up the output
#         translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(f"Raw translated text: {translated_text}")
#
#         # Remove the prompt text from the beginning, if present
#         if "[INST]" in translated_text and "[/INST]" in translated_text:
#             translated_text = translated_text.split("[/INST]")[1].strip()
#
#         # Log the cleaned translated text
#         print(f"Cleaned translated text: {translated_text}")
#         max_retries = 3  # Maximum number of retry attempts
#         attempt = 0
#         while attempt < max_retries:
#             # Identify sentences with Hebrew characters
#             sentences = re.split(r'(?<=[.!?]) +', translated_text)
#             hebrew_sentences = [s for s in sentences if re.search(r'[א-ת]', s)]
#
#             if not hebrew_sentences:
#                 break  # No Hebrew sentences left
#
#             print(f"Attempt {attempt + 1}: Translating {len(hebrew_sentences)} Hebrew sentence(s).")
#
#             # Translate each Hebrew sentence individually
#             for hebrew_sentence in hebrew_sentences:
#                 print(f"Translating sentence: {hebrew_sentence}")
#                 translated_sentence = translate_text(hebrew_sentence)
#                 print(f"Translated sentence: {translated_sentence}")
#
#                 # Replace the Hebrew sentence with its translation
#                 translated_text = translated_text.replace(hebrew_sentence, translated_sentence)
#
#             attempt += 1
#
#         # Final check after retries
#         if re.search(r'[א-ת]', translated_text):
#             print("Error: Translation still contains Hebrew characters after maximum retries.")
#             # Optionally, you can remove remaining Hebrew characters
#             translated_text = re.sub(r'[א-ת]', '', translated_text)
#             print(f"Final post-processed translated text: {translated_text}")
#         # *** End of Changes ***
#
#         return translated_text
#     except Exception as e:
#         print(f"Error during translation: {e}")
#         return "Translation Error"
#
#     #     # Post-process to handle any remaining Hebrew text
#     #     if re.search(r'[א-ת]', translated_text):
#     #         print("Warning: Translated text still contains Hebrew characters.")
#     #         # Optionally, you can replace or flag the Hebrew text
#     #         translated_text = re.sub(r'[א-ת]', '', translated_text)
#     #         print(f"Post-processed translated text: {translated_text}")
#     #
#     #     return translated_text
#     # except Exception as e:
#     #     print(f"Error during translation: {e}")
#     #     return "Translation Error"
