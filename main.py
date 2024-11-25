
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

# from ML-CLASSIFICATION-ASSISTANT import classification, eda

import classification, eda

# Sidebar Navigation
st.sidebar.title("Data Analysis")
page = st.sidebar.radio("Select a Page", ["EDA", "Classification"])

# Load the selected page
if page == "EDA":
    eda.app()
elif page == "Classification":
    classification.app()
