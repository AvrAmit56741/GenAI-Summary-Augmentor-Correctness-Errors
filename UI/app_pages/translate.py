# import streamlit as st
# import json
# st.set_page_config(layout="wide")
#
# def load_jsonl(file):
#     """Load a JSONL file and handle errors gracefully."""
#     try:
#         return [json.loads(line) for line in file]
#     except json.JSONDecodeError as e:
#         st.error(f"Error parsing JSONL: {e}")
#     except Exception as e:
#         st.error(f"Unexpected error: {e}")
#     return None
#
# def format_field_content(content):
#     """Format field content for display."""
#     if isinstance(content, (dict, list)):
#         return json.dumps(content, indent=4)
#     return str(content)
#
# def calculate_dynamic_height(content, min_height=400, max_height=800):
#     """Calculate the dynamic height of the text area based on content length."""
#     line_count = len(content.splitlines())
#     return min(max(min_height, line_count * 20), max_height)
#
# def display_comparison(data, field1, field2):
#     """Display a side-by-side comparison of two JSON fields."""
#     if field1 not in data or field2 not in data:
#         st.warning("Selected fields are not present in the JSONL element.")
#         return
#
#     st.write(f"### Comparison of {field1} and {field2}")
#
#     # Prepare content for display
#     field1_content = format_field_content(data[field1])
#     field2_content = format_field_content(data[field2])
#
#     # Adjust text area size
#     height = 800  # Increased height
#
#     # Use st.container for full-page width layout
#     with st.container():
#         col1, col2 = st.columns([5, 5])  # Maximize column width
#         with col1:
#             st.write(f"**{field1}:**")
#             st.text_area("", field1_content, height=height, label_visibility="collapsed")
#         with col2:
#             st.write(f"**{field2}:**")
#             st.text_area("", field2_content, height=height, label_visibility="collapsed")
#
# def main():
#     """Main function for the Streamlit app."""
#     st.title("JSONL Field Comparison App")
#
#     uploaded_file = st.file_uploader("Upload a JSONL file", type="jsonl")
#     if uploaded_file is not None:
#         data = load_jsonl(uploaded_file)
#         if data:
#             # Extract doc_id from metadata if present
#             doc_ids = [str(item.get("metadata", {}).get("doc_id", "Unknown")) for item in data]
#
#             # Ensure there are at least one doc_id to select
#             if not doc_ids:
#                 st.warning("The uploaded JSONL must have at least one element with a 'doc_id' field.")
#                 return
#
#             # Dropdown for doc_id selection
#             selected_doc_id = st.selectbox("Select a doc_id", doc_ids, key="doc_id")
#
#             # Find the selected document
#             selected_doc = next((item for item in data if str(item.get("metadata", {}).get("doc_id")) == selected_doc_id), None)
#
#             if selected_doc:
#                 fields = [key for key in selected_doc.keys() if key != "metadata"]
#
#                 # Ensure there are at least two fields to compare
#                 if len(fields) < 2:
#                     st.warning("The selected document must have at least two fields for comparison.")
#                     return
#
#                 # Dropdowns for field selection
#                 field1 = st.selectbox("Select first field to compare", fields, key="field1")
#                 field2 = st.selectbox("Select second field to compare", fields, key="field2")
#
#                 if field1 and field2 and field1 != field2:
#                     display_comparison(selected_doc, field1, field2)
#                 elif field1 == field2:
#                     st.warning("Please select two different fields to compare.")
#
# def app():
#     """Function to integrate this page into a multi-page app."""
#     main()
#
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import json
#
#
# def load_jsonl(file):
#     """Load a JSONL file and handle errors gracefully."""
#     try:
#         return [json.loads(line) for line in file]
#     except json.JSONDecodeError as e:
#         st.error(f"Error parsing JSONL: {e}")
#     except Exception as e:
#         st.error(f"Unexpected error: {e}")
#     return None
#
# def format_field_content(content):
#     """Format field content for display."""
#     if isinstance(content, (dict, list)):
#         return json.dumps(content, indent=4)
#     return str(content)
#
# def calculate_dynamic_height(content, min_height=400, max_height=800):
#     """Calculate the dynamic height of the text area based on content length."""
#     line_count = len(content.splitlines())
#     return min(max(min_height, line_count * 20), max_height)
#
# def display_comparison(data, field1, field2):
#     """Display a side-by-side comparison of two JSON fields."""
#     if field1 not in data or field2 not in data:
#         st.warning("Selected fields are not present in the JSONL element.")
#         return
#
#     st.write(f"### Comparison of {field1} and {field2}")
#
#     # Prepare content for display
#     field1_content = format_field_content(data[field1])
#     field2_content = format_field_content(data[field2])
#
#     # Adjust text area size
#     height = 800  # Increased height
#
#     # Use st.container for full-page width layout
#     with st.container():
#         col1, col2 = st.columns([5, 5])  # Maximize column width
#         with col1:
#             st.write(f"**{field1}:**")
#             st.text_area("", field1_content, height=height, label_visibility="collapsed")
#         with col2:
#             st.write(f"**{field2}:**")
#             st.text_area("", field2_content, height=height, label_visibility="collapsed")
#
# def app():
#     """Main function for the Compare Fields page."""
#     st.title("JSONL Field Comparison App")
#
#     uploaded_file = st.file_uploader("Upload a JSONL file", type="jsonl")
#     if uploaded_file is not None:
#         data = load_jsonl(uploaded_file)
#         if data:
#             # Extract doc_id from metadata if present
#             doc_ids = [str(item.get("metadata", {}).get("doc_id", "Unknown")) for item in data]
#
#             # Ensure there are at least one doc_id to select
#             if not doc_ids:
#                 st.warning("The uploaded JSONL must have at least one element with a 'doc_id' field.")
#                 return
#
#             # Dropdown for doc_id selection
#             selected_doc_id = st.selectbox("Select a doc_id", doc_ids, key="doc_id")
#
#             # Find the selected document
#             selected_doc = next((item for item in data if str(item.get("metadata", {}).get("doc_id")) == selected_doc_id), None)
#
#             if selected_doc:
#                 fields = [key for key in selected_doc.keys() if key != "metadata"]
#
#                 # Ensure there are at least two fields to compare
#                 if len(fields) < 2:
#                     st.warning("The selected document must have at least two fields for comparison.")
#                     return
#
#                 # Dropdowns for field selection
#                 field1 = st.selectbox("Select first field to compare", fields, key="field1")
#                 field2 = st.selectbox("Select second field to compare", fields, key="field2")
#
#                 if field1 and field2 and field1 != field2:
#                     display_comparison(selected_doc, field1, field2)
#                 elif field1 == field2:
#                     st.warning("Please select two different fields to compare.")
#
# if __name__ == "__main__":
#     app()

import streamlit as st
import json


def load_jsonl(file):
    """Load a JSONL file and handle errors gracefully."""
    try:
        return [json.loads(line) for line in file]
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSONL: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None

def format_field_content(content):
    """Format field content for display."""
    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=4)
    return str(content)

def calculate_dynamic_height(content, min_height=400, max_height=800):
    """Calculate the dynamic height of the text area based on content length."""
    line_count = len(content.splitlines())
    return min(max(min_height, line_count * 20), max_height)

def display_comparison(data, field1, field2):
    """Display a side-by-side comparison of two JSON fields."""
    if field1 not in data or field2 not in data:
        st.warning("Selected fields are not present in the JSONL element.")
        return

    st.write(f"### Comparison of {field1} and {field2}")

    # Prepare content for display
    field1_content = format_field_content(data[field1])
    field2_content = format_field_content(data[field2])

    # Adjust text area size
    height = 800  # Increased height

    # Use st.container for full-page width layout
    with st.container():
        col1, col2 = st.columns([5, 5])  # Maximize column width
        with col1:
            st.write(f"**{field1}:**")
            st.text_area("", field1_content, height=height, label_visibility="collapsed")
        with col2:
            st.write(f"**{field2}:**")
            st.text_area("", field2_content, height=height, label_visibility="collapsed")

def app():
    """Main function for the Compare Fields page."""
    st.title("JSONL Field Comparison App")

    uploaded_file = st.file_uploader("Upload a JSONL file", type="jsonl")
    if uploaded_file is not None:
        data = load_jsonl(uploaded_file)
        if data:
            # Extract doc_id from metadata if present
            doc_ids = [str(item.get("metadata", {}).get("doc_id", "Unknown")) for item in data]

            # Ensure there are at least one doc_id to select
            if not doc_ids:
                st.warning("The uploaded JSONL must have at least one element with a 'doc_id' field.")
                return

            # Dropdown for doc_id selection
            selected_doc_id = st.selectbox("Select a doc_id", doc_ids, key="doc_id")

            # Find the selected document
            selected_doc = next(
                (item for item in data if str(item.get("metadata", {}).get("doc_id")) == selected_doc_id), None)

            if selected_doc:
                fields = [key for key in selected_doc.keys() if key != "metadata"]

                # Ensure there are at least two fields to compare
                if len(fields) < 2:
                    st.warning("The selected document must have at least two fields for comparison.")
                    return

                # Dropdowns for field selection
                field1 = st.selectbox("Select first field to compare", fields, key="field1")
                field2 = st.selectbox("Select second field to compare", fields, key="field2")

                if field1 and field2 and field1 != field2:
                    display_comparison(selected_doc, field1, field2)
                elif field1 == field2:
                    st.warning("Please select two different fields to compare.")

                # Display only the qafacteval_results
                st.write("### QA FactEval Results:")
                qafacteval_results = selected_doc.get("qafacteval_results", "No qafacteval_results found.")
                st.json(qafacteval_results)  # Streamlit's built-in JSON viewer

if __name__ == "__main__":
    app()
