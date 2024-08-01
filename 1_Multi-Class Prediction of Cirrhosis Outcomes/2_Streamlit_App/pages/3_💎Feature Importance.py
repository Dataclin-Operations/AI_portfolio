# #==========================================================================#
# #                          Import Libiraries
# #==========================================================================#
import sys
import streamlit as st
import pandas as pd
import os
import time

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



import shap

import warnings
# Suppress specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)




#==========================================================================#
#                          Import Data and functions From main file
#==========================================================================#

sys.path.append(os.path.abspath('../functions.py'))

from functions import test_X, ml_model





#==========================================================================#
#                          Shap Values
#==========================================================================#
@st.experimental_fragment()
@st.cache_data()
def shap_explainer_chart(_selected_model, chart_type, number):

    start_time = time.time()
    explainer = shap.TreeExplainer(selected_model)

    # calculate shap values. This is what we will plot.
    # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
    shap_values = explainer.shap_values(test_X)
    plt.figure(figsize=(4, 2))
    if chart_type =="Bar":
        col1, col2, col3, col4, col5 = st.columns(5)
        with col2:
            st.markdown("<h5 style='color: #008080; text-align:center'>Class Mapping</h5>", unsafe_allow_html=True)
        with col3:
            st.markdown("<h5 style='color: #008080; text-align:center'>Class 0: <b style='color:red'>C</b></h5>", unsafe_allow_html=True)
        with col4:
            st.markdown("<h5 style='color: #008080; text-align:center'>Class 1: <b style='color:red'>CL</b></h5>", unsafe_allow_html=True)
        with col5:
            st.markdown("<h5 style='color: #008080; text-align:center'>Class 2: <b style='color:red'>D</b></h5>", unsafe_allow_html=True)
        shap.summary_plot(shap_values, test_X, max_display=20)
        st.pyplot(plt.gcf())
    else:
        shap.summary_plot(shap_values[1], test_X, show=False)
        st.pyplot(plt.gcf())

    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    return execution_time
    
#==========================================================================#
#                          Main Application
#==========================================================================#


if __name__ == "__main__":
    logo_path = "./TheLogo2.png"  # Replace with the path to your logo image
    st.sidebar.image(logo_path, use_column_width=True)
    st.sidebar.markdown("<h2 style='color: #008080; text-align:center'>Select a Model</h2>", unsafe_allow_html=True)
    # col1, col2, col3 = st.columns(3)
    # with col1:
    model_select2 = st.sidebar.selectbox(" ", ["LightGBM", "XGBoost"], label_visibility="collapsed", key="second_select") 
    tabs = st.tabs(["Summary Plot", "Bar Plot"])
    with tabs[0]:
        if model_select2 == "XGBoost":
            all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("xgb")
            exe_time = shap_explainer_chart(selected_model, "Summary", 1)
            st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")
        elif model_select2 == "LightGBM":
            all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("LighGBM")
            exe_time = shap_explainer_chart(selected_model, "Summary", 2)
            st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")
        # elif model_select2 == "Randomforest":
        #     all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("rf")
        #     exe_time = shap_explainer_chart(selected_model, "Summary", 3)
        #     st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")
    with tabs[1]:
        if model_select2 == "XGBoost":
            all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("xgb")
            exe_time = shap_explainer_chart(selected_model, "Bar", 4)
            # st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")
        elif model_select2 == "LightGBM":
            all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("LighGBM")
            exe_time = shap_explainer_chart(selected_model, "Bar", 5)
            # st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")
        # elif model_select2 == "Randomforest":
        #     all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("rf")
        #     exe_time = shap_explainer_chart(selected_model, "Bar", 6)
        #     st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")
    
   

