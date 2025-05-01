# import streamlit as st
# from app_pages.qafacteval import app as qafacteval_app
# from app_pages.translate import app as translate_app
#
# # Set page configuration only once
# st.set_page_config(page_title="JSONL Viewer", layout="wide")
#
# def main():
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Go to", ["Compare Fields", "QA FactEval Viewer"])
#
#     if page == "Compare Fields":
#         translate_app()  # Call the app from translate.py
#     elif page == "QA FactEval Viewer":
#         qafacteval_app()  # Call the app from qafacteval.py
#
# if __name__ == "__main__":
#     main()
# main.py

import streamlit as st
from app_pages.qafacteval import app as qafacteval_app
from app_pages.translate import app as translate_app
from app_pages.dae import app as evaluate_app  # Import the new evaluate app
from app_pages.corruption_augmentation import app as augmentation_app
from app_pages.factcc.data_augmentation_page import app as data_augmentation_app

# Set page configuration only once
st.set_page_config(page_title="JSONL Viewer", layout="wide")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Compare Fields", "QA FactEval Metric", "DAE Metric","Text Augmentation - Hebrew","FactCC Augmentation - English"])

    if page == "Compare Fields":
        translate_app()  # Call the app from translate.py
    elif page == "QA FactEval Metric":
        qafacteval_app()  # Call the app from qafacteval.py
    elif page == "DAE Metric":
        evaluate_app()  # Call the new evaluate app
    elif page == "Text Augmentation - Hebrew":
        augmentation_app()  # Call the augmentation app
    elif page == "FactCC Augmentation - English":
        data_augmentation_app()  # Call the new data augmentation app


if __name__ == "__main__":
    import streamlit as st
    main()
