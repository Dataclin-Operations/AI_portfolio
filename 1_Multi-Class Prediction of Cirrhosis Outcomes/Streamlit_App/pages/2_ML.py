# #==========================================================================#
# #                          Import Libiraries
# #==========================================================================#
import sys
from streamlit_extras.colored_header import colored_header 
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
from st_aggrid.shared import GridUpdateMode
from streamlit_extras.dataframe_explorer import dataframe_explorer 
import os
import yaml
import time

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.express as px
import plotly.graph_objects as go
import datetime



# import machine learning libiraries
from sklearn.metrics import make_scorer, confusion_matrix, precision_score, recall_score, accuracy_score, classification_report, f1_score,roc_auc_score, roc_curve, auc, log_loss

from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize, OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

import xgboost as xgb
from lightgbm import LGBMClassifier 
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


import re
import shap
import optuna

import warnings
# Suppress specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)



#==========================================================================#
#                          Import Data and functions From main file
#==========================================================================#

sys.path.append(os.path.abspath('../functions.py'))

from functions import  ml_model, test_X





def disply_test_train(test_train, name):
    # Display the DataFrame
    st.markdown(f"<h4 style='color: #008080; text-align:center'>{name} Train vs Test Scores</h4>", unsafe_allow_html=True)
    

    gd = GridOptionsBuilder.from_dataframe(test_train)
    gd.configure_column("Test_Score", header_name="Test_Score",groupable=False,filter=True,autoSize=True,
                    resizable=True,
                    # headerStyle={'textAlign': 'center', 'fontSize': '50px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': '#008080'},  # Styling for header
                    cellStyle={'textAlign': 'center', 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif','color': 'red'}
                    
                        )
    gd.configure_column("Train_Score", header_name="Train_Score",minWidth=250,groupable=False,filter=True,autoSize=True,
                    resizable=True,
                    # headerStyle={'textAlign': 'center', 'fontSize': '50px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': '#008080'},  # Styling for header
                    cellStyle={'textAlign': 'center', 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif','color': '#008080'}
                    
                        )
    gd.configure_column("Metric", header_name="Metric",minWidth=550,groupable=True,filter=True,autoSize=False,
                resizable=True,
                # headerStyle={'textAlign': 'center', 'fontSize': '50px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': '#008080'},  # Styling for header
                cellStyle={'textAlign': 'center', 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif','color': '#008080'}
                
                    )
    gd.configure_default_column(
                    filter=True,autoSize=False,
                    resizable=True,
                    # headerStyle={'textAlign': 'center', 'fontSize': '50px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': '#008080'},  # Styling for header
                    cellStyle={'textAlign': 'center', 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif','color': '#008080'}
                    )
    gridoptions = gd.build()
    # Display the custom CSS
    
    grid_table = AgGrid(test_train.reset_index(),gridOptions=gridoptions,
                update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                height = 160,
                allow_unsafe_jscode=True,
                enable_enterprise_modules = True,
                theme = 'alpine',columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
        






def disply_scores(df_scores, name):
    # Display the DataFrame
    st.markdown(f"<h4 style='color: #008080; text-align:center'>{name} Scores</h4>", unsafe_allow_html=True)
    
    # col1, col2 = st.columns(2)
    # with col1:
    gd = GridOptionsBuilder.from_dataframe(df_scores)
    gd.configure_column("Score", header_name="Score",minWidth=20,groupable=False,filter=True,autoSize=True,
                    resizable=True,
                    # headerStyle={'textAlign': 'center', 'fontSize': '50px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': '#008080'},  # Styling for header
                    cellStyle={'textAlign': 'left', 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif','color': '#008080'}
                    
                        )
    gd.configure_column("Metric", header_name="Metric",minWidth=50,groupable=True,filter=True,autoSize=False,
                    resizable=True,
                    # headerStyle={'textAlign': 'center', 'fontSize': '50px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': '#008080'},  # Styling for header
                    cellStyle={'textAlign': 'left', 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif','color': '#008080'}
                    
                        )
    gd.configure_default_column(
                    filter=True,autoSize=False,
                    resizable=True,
                    # headerStyle={'textAlign': 'center', 'fontSize': '50px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': '#008080'},  # Styling for header
                    cellStyle={'textAlign': 'left', 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif','color': '#008080'}
                    )
    gridoptions = gd.build()
    # Display the custom CSS
    
    grid_table = AgGrid(df_scores.reset_index(),gridOptions=gridoptions,
                update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                height = 290,
                allow_unsafe_jscode=True,
                enable_enterprise_modules = True,
                theme = 'alpine',columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
      
if __name__ == "__main__":
    logo_path = "./TheLogo2.png"  # Replace with the path to your logo image
    st.sidebar.image(logo_path, use_column_width=True)
    st.sidebar.markdown("<h2 style='color: #008080; text-align:center'>Select a Model</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        model_select = st.sidebar.selectbox(" ", ["LightGBM", "XGBoost", "Randomforest"], label_visibility="collapsed")    
    if model_select == "XGBoost":
        all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("xgb")
    elif model_select == "LightGBM":
        all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("LighGBM")
    elif model_select == "Randomforest":
        all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("rf")
    
    disply_scores(all_scores_df2, model_select)
    disply_test_train(train_test_scores_df2, model_select)
   
    st.sidebar.success(f"The Model execution time is: {ex_time:.2f} seconds")