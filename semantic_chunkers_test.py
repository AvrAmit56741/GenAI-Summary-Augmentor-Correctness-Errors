import json
from semantic_router.encoders import HuggingFaceEncoder
from semantic_chunkers import ConsecutiveChunker
from semantic_chunkers import StatisticalChunker
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


# --------------------------
# 1. Initialize Encoder and Chunker
# --------------------------

def initialize_chunker():
    """
    Initializes the HuggingFaceEncoder and ConsecutiveChunker.
    Returns:
        ConsecutiveChunker: An instance of ConsecutiveChunker.
    """
    # Specify the Hugging Face model for encoding
    encoder_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    logging.info(f"Initializing HuggingFaceEncoder with model '{encoder_model_name}'...")

    try:
        encoder = HuggingFaceEncoder(model_name=encoder_model_name)
        logging.info("HuggingFaceEncoder initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing HuggingFaceEncoder: {e}")
        raise e

    # Initialize the ConsecutiveChunker with the encoder and a similarity threshold
    similarity_threshold = 0.3  # Adjust based on your data and requirements
    try:
        #chunker = ConsecutiveChunker(encoder=encoder, score_threshold=similarity_threshold)
        chunker = StatisticalChunker(encoder=encoder)
        logging.info("ConsecutiveChunker initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing ConsecutiveChunker: {e}")
        raise e

    return chunker


# --------------------------
# 2. Define the Process Function
# --------------------------

def process_jsonl(input_path, output_path, chunker, max_lines=20):
    """
    Processes a JSONL file, performs semantic chunking on the 'text_raw' field,
    prints the chunks, and writes the updated JSON objects to a new JSONL file.

    Args:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to the output JSONL file.
        chunker (ConsecutiveChunker): An instance of ConsecutiveChunker.
        max_lines (int): Maximum number of lines to process.
    """
    logging.info(f"Processing input file: {input_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:

            line_count = 0
            processed_count = 0

            for line in infile:
                line_count += 1
                if processed_count >= max_lines:
                    logging.info(f"Processed {max_lines} lines, stopping further processing.")
                    break

                logging.info(f"Processing line {line_count}: {line.strip()}")

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error on line {line_count}: {e}")
                    continue  # Skip malformed JSON lines

                # Perform semantic chunking on 'text_raw' if it exists
                if "text_raw" in data:
                    text = data["text_raw"]
                    logging.info(f"Performing semantic chunking on 'text_raw' field.")

                    try:
                        # The chunker expects a list of documents; here, it's a single document
                        chunks = chunker(docs=[text])
                        text_chunks = chunks[0]  # Extract the list of chunks for the first document

                        # Extract text from each Chunk object
                        extracted_chunks = [" ".join(chunk.splits) for chunk in text_chunks]
                        data["text_raw_chunks"] = extracted_chunks

                        # Print the chunks
                        logging.info(f"Generated {len(extracted_chunks)} chunks for line {line_count}:")
                        for idx, chunk_text in enumerate(extracted_chunks, start=1):
                            print(f"Line {line_count} - Chunk {idx}: {chunk_text}")

                    except Exception as e:
                        logging.error(f"Error during chunking on line {line_count}: {e}")
                        data["text_raw_chunks"] = []

                # Write the updated JSON object to the output file
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                logging.info(f"Processed line {line_count} successfully.")
                processed_count += 1

        logging.info(f"Processing completed. Output saved to {output_path}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred while processing the file: {e}")


# --------------------------
# 3. Main Execution
# --------------------------

if __name__ == "__main__":
    # Initialize the ConsecutiveChunker
    chunker = initialize_chunker()

    # Define paths to input and output JSONL files
    input_jsonl_path = "/sise/home/avramit/pycharm_project_502/summarization-7-heb_up.jsonl"  # Replace with your input path
    output_jsonl_path = "/sise/home/avramit/pycharm_project_502/output_new_2.jsonl"  # Replace with your output path

    # Process the JSONL file
    process_jsonl(input_jsonl_path, output_jsonl_path, chunker, max_lines=20)
