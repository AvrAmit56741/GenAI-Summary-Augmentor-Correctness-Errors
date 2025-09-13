import sys
import os

# Add the parent directory to the Python path so we can import model_analytics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_analytics import model_analytics_page

# Configure the page
import streamlit as st
st.set_page_config(page_title="Model Analytics", layout="wide", page_icon="🔬")

# Run the Model Analytics page
model_analytics_page() 