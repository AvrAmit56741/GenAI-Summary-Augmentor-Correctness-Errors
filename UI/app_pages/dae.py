# import streamlit as st
# import subprocess
# import tempfile
# import os
# import json
#
# def app():
#     st.title("Evaluate Generated Outputs")
#
#     st.markdown("""
#     This page allows you to evaluate summaries using the `evaluate_generated_outputs.py` script.
#     Provide the input manually or upload a JSON file in the required format. The script will generate results interactively.
#     """)
#
#     # Input method selection
#     # Input method selection
#     input_method = st.radio("Input Method", ["Manual Input", "Upload JSONL File"])
#
#     # Placeholder variables
#     source_document = ""
#     summary = ""
#
#     if input_method == "Manual Input":
#         st.subheader("Manual Input")
#         source_document = st.text_area("Source Document", height=200)
#         summary = st.text_area("Summary", height=150)
#     elif input_method == "Upload JSONL File":
#         st.subheader("Upload JSONL File")
#         uploaded_jsonl = st.file_uploader("Upload JSONL File", type=["jsonl"])
#
#         if uploaded_jsonl:
#             try:
#                 # Decode binary content and parse JSONL
#                 content = uploaded_jsonl.read().decode("utf-8")  # Read and decode binary content
#                 data = [json.loads(line) for line in content.strip().split("\n")]  # Parse JSONL lines
#
#                 doc_ids = [doc.get("metadata", {}).get("doc_id", f"doc_{i}") for i, doc in enumerate(data)]
#                 selected_doc_id = st.selectbox("Select Document ID", doc_ids)
#
#                 # Find the selected document
#                 selected_doc = next((doc for doc in data if doc.get("metadata", {}).get("doc_id") == selected_doc_id),
#                                     None)
#                 if selected_doc:
#                     keys = list(selected_doc.keys())
#                     text_key = st.selectbox("Select Text Key (Source Document)", keys)
#                     summary_key = st.selectbox("Select Summary Key", keys)
#
#                     source_document = selected_doc.get(text_key, "")
#                     summary = selected_doc.get(summary_key, "")
#
#                     st.text_area("Source Document", source_document, height=200, disabled=True)
#                     st.text_area("Summary", summary, height=150, disabled=True)
#
#             except Exception as e:
#                 st.error(f"Error processing JSONL file: {e}")
#
#     # elif input_method == "Upload JSONL File":
#     #     st.subheader("Upload JSONL File")
#     #     uploaded_jsonl = st.file_uploader("Upload JSONL File", type=["jsonl"])
#     #
#     #     if uploaded_jsonl:
#     #         try:
#     #             # Load and parse the JSONL file
#     #             data = [json.loads(line) for line in uploaded_jsonl]
#     #             doc_ids = [doc.get("metadata", {}).get("doc_id", f"doc_{i}") for i, doc in enumerate(data)]
#     #             selected_doc_id = st.selectbox("Select Document ID", doc_ids)
#     #
#     #             # Find the selected document
#     #             selected_doc = next((doc for doc in data if doc.get("metadata", {}).get("doc_id") == selected_doc_id), None)
#     #             if selected_doc:
#     #                 keys = list(selected_doc.keys())
#     #                 text_key = st.selectbox("Select Text Key (Source Document)", keys)
#     #                 summary_key = st.selectbox("Select Summary Key", keys)
#     #
#     #                 source_document = selected_doc.get(text_key, "")
#     #                 summary = selected_doc.get(summary_key, "")
#     #
#     #                 st.text_area("Source Document", source_document, height=200, disabled=True)
#     #                 st.text_area("Summary", summary, height=150, disabled=True)
#     #
#     #         except Exception as e:
#     #             st.error(f"Error processing JSONL file: {e}")
#
#     # Run evaluation when "Evaluate" button is clicked
#     if st.button("Evaluate"):
#         if not source_document.strip() or not summary.strip():
#             st.error("Both source document and summary are required.")
#             return
#
#         # Write input data to a temporary file
#         with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
#             temp_file.write(f"{source_document.strip()}\n{summary.strip()}\n\n")
#             temp_file_path = temp_file.name
#
#         # Define script command
#         model_type = "electra_dae"
#         model_dir = "/home/avramit/factuality-datasets/corenlp/stanford-corenlp-full-2018-02-27/DAE"
#         script_path =  "/home/avramit/factuality-datasets/evaluate_generated_outputs.py"
#
#         command = [
#             "conda", "run", "-n", "dae_factuality", "python3", script_path,
#             "--model_type", model_type,
#             "--model_dir", model_dir,
#             "--input_file", temp_file_path
#         ]
#
#         # Execute the script and display output
#         with st.spinner("Running evaluation..."):
#             try:
#                 result = subprocess.run(command, text=True, capture_output=True, check=True)
#                 st.text("### Script Output")
#                 st.text(result.stdout)
#
#                 if result.stderr:
#                     st.text("### Script Errors")
#                     st.text(result.stderr)
#             except subprocess.CalledProcessError as e:
#                 st.error("Error during script execution.")
#                 st.text(e.stderr)
#             finally:
#                 # Clean up temporary file
#                 os.unlink(temp_file_path)
import streamlit as st
import subprocess
import tempfile
import os
import json

def app():
    st.title("Evaluate Generated Outputs")

    st.markdown("""
    This page allows you to evaluate summaries using the `evaluate_generated_outputs.py` script.
    Provide the input manually or upload a JSONL file in the required format. The script will generate results interactively.
    """)

    # Input method selection
    input_method = st.radio("Input Method", ["Manual Input", "Upload JSONL File"])

    # Placeholder variables
    source_document = ""
    summary = ""

    if input_method == "Manual Input":
        st.subheader("Manual Input")
        source_document = st.text_area("Source Document", height=200)
        summary = st.text_area("Summary", height=150)

    elif input_method == "Upload JSONL File":
        st.subheader("Upload JSONL File")
        uploaded_jsonl = st.file_uploader("Upload JSONL File", type=["jsonl"])

        if uploaded_jsonl:
            try:
                # Decode binary content and parse JSONL
                content = uploaded_jsonl.read().decode("utf-8")  # Read and decode binary content
                data = [json.loads(line) for line in content.strip().split("\n") if line.strip()]  # Parse JSONL lines

                if not data:
                    st.error("The uploaded JSONL file is empty or improperly formatted.")
                else:
                    # Extract doc_ids, ensuring uniqueness
                    doc_ids = [doc.get("metadata", {}).get("doc_id", f"doc_{i}") for i, doc in enumerate(data)]
                    selected_doc_id = st.selectbox("Select Document ID", doc_ids)

                    # Find the selected document
                    selected_doc = next((doc for doc in data if doc.get("metadata", {}).get("doc_id") == selected_doc_id), None)
                    if selected_doc:
                        # Exclude 'metadata' from key selection and include only string fields
                        string_keys = [key for key in selected_doc.keys() if isinstance(selected_doc[key], str)]
                        if len(string_keys) < 2:
                            st.error("The selected document does not have enough string fields for source document and summary.")
                        else:
                            # Ensure that text_key and summary_key are distinct
                            text_key = st.selectbox("Select Text Key (Source Document)", string_keys, key="text_key")
                            summary_key = st.selectbox("Select Summary Key", [key for key in string_keys if key != text_key], key="summary_key")

                            source_document = selected_doc.get(text_key, "")
                            summary = selected_doc.get(summary_key, "")

                            if not source_document.strip() or not summary.strip():
                                st.error("Selected source document or summary is empty.")
                            else:
                                # Replace internal newlines with spaces to ensure correct format
                                source_clean = source_document.strip().replace("\n", " ")
                                summary_clean = summary.strip().replace("\n", " ")

                                # Display the selected fields
                                st.text_area("Source Document", source_clean, height=200, disabled=True)
                                st.text_area("Summary", summary_clean, height=150, disabled=True)

            except json.JSONDecodeError:
                st.error("Invalid JSONL file. Please ensure each line is a valid JSON object.")
            except Exception as e:
                st.error(f"Error processing JSONL file: {e}")

    # Run evaluation when "Evaluate" button is clicked
    if st.button("Evaluate"):
        if not source_document.strip() or not summary.strip():
            st.error("Both source document and summary are required.")
            return

        # Write input data to a temporary file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as temp_file:
            # Replace internal newlines with spaces to prevent format issues
            source_clean = source_document.strip().replace("\n", " ")
            summary_clean = summary.strip().replace("\n", " ")
            temp_file.write(f"{source_clean}\n{summary_clean}\n\n")
            temp_file_path = temp_file.name

        # Define script command
        model_type = "electra_dae"
        model_dir = "/home/avramit/factuality-datasets/corenlp/stanford-corenlp-full-2018-02-27/DAE"
        script_path = "/home/avramit/factuality-datasets/evaluate_generated_outputs.py"

        command = [
            "conda", "run", "-n", "dae_factuality", "python3", script_path,
            "--model_type", model_type,
            "--model_dir", model_dir,
            "--input_file", temp_file_path
        ]

        # Execute the script and display output
        with st.spinner("Running evaluation..."):
            try:
                result = subprocess.run(command, text=True, capture_output=True, check=True)
                st.text("### Script Output")
                st.text(result.stdout)

                if result.stderr:
                    st.text("### Script Errors")
                    st.text(result.stderr)
            except subprocess.CalledProcessError as e:
                st.error("Error during script execution.")
                st.text(e.stderr)
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
