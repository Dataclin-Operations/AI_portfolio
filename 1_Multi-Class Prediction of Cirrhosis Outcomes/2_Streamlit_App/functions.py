#==========================================================================#
#                          Import Libiraries
#==========================================================================#
import sys
# Caution: path[0] is reserved for script path (or '' in REPL)
from streamlit_extras.colored_header import colored_header 
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
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


import re
import shap
import optuna

import warnings
# Suppress specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)




#==========================================================================#
#                          Charts Functions
#==========================================================================#



def bar_chart(df, xx, yy,  txt="", labls={},hover={}, colors=[], hight=600, wdth=900, ttle="ttle",xtitle="xtitle" , ytitle="ytitle" ,colour=None, bg=0.6, bgg=0.1, leg=None, box=False,yscale=False,yscale_percentage=False, start=-1, end=1):
    if box:
        fig = px.box(df,  x=xx, y=yy, color=colour,
            color_discrete_sequence=colors,height=hight, width=wdth
            )
    elif colour ==None:
        fig = px.bar(df, x=xx, y=yy, barmode='group',text=txt,
             labels=labls,
             hover_data=hover,
             title=ttle,
            color_discrete_sequence=colors,
                     height=hight, width=wdth)
    else:
        fig = px.bar(df, x=xx, y=yy, color=colour, barmode='group',text=txt,
                     labels=labls,
                     hover_data=hover,
                     title=ttle,
                     color_discrete_sequence=colors,height=hight, width=wdth)


    if box == False:
        fig.update_traces(textfont_size=15, textposition='inside')
    
    fig.update_layout(
        legend_title_text=leg,
        #background 
        # plot_bgcolor="white",  #background color
        bargap=bg,  # Gap between bars
        bargroupgap=bgg, 
    
        # Title
         title={'text': ttle, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, #title center and size
         title_font_size=22,  # Title font size
    
        # Xaxis
        xaxis=dict(
            title=xtitle,  # Example X axis title
            titlefont_size=18,  # Size of the X axis title font
            tickfont_size=12,   # Size of the X axis tick labels font
            showline=True,  # Show x-axis line
            showgrid=False, # showgrid
            gridcolor="lightgray", #grid color
            
        ),
        yaxis=dict(
            title=ytitle,  # Example Y axis title
            titlefont_size=18,  # Size of the Y axis title font
            tickfont_size=12,   # Size of the Y axis tick labels font
         
            showgrid=True, # showgrid
            
        ),

    legend=dict(
                font_size=14,  # Size of the legend font
                
            ),
        # font
        font=dict(
            family='Arial', size=14  # Overall font size
        )
    )
    if yscale_percentage and yscale:
        fig.update_xaxes(title_text=xtitle, # Y axisTitle
                         zeroline=True,
                         showline=True,  # Show y-axis line
                        linecolor='gray',
                        titlefont_size=18,  # Set the title font size (optional)
                        titlefont_color="#008080",  # Set the title font color (optional)
                         )
        fig.update_yaxes(title_text=ytitle, # Y axisTitle
                         range=[start, end],
                         tickformat='.0%',
                         zeroline=True,
                         showline=True,  # Show y-axis line
                         linecolor='gray',  # Color of the y-axis line
                         titlefont_size=18,  # Set the title font size (optional)
                        titlefont_color="#008080",  # Set the title font color (optional)         )
                        )
    elif yscale_percentage:
        fig.update_xaxes(title_text=xtitle, # Y axisTitle
                         zeroline=True,
                         showline=True,  # Show y-axis line
                        linecolor='gray',
                        titlefont_size=18,  # Set the title font size (optional)
                        titlefont_color="#008080",  # Set the title font color (optional)
                         )
        fig.update_yaxes(title_text=ytitle, # Y axisTitle
                         zeroline=True,
                         tickformat='.0%',
                         showline=True,  # Show y-axis line
                         linecolor='gray',  # Color of the y-axis line
                         titlefont_size=16,  # Set the title font size (optional)
                        titlefont_color="#008080",  # Set the title font color (optional)         )
                        )
    elif yscale:
        fig.update_xaxes(title_text=xtitle, # Y axisTitle
                         zeroline=True,
                         showline=True,  # Show y-axis line
                        linecolor='gray',
                        titlefont_size=18,  # Set the title font size (optional)
                        titlefont_color="#008080",  # Set the title font color (optional)
                         )
        fig.update_yaxes(title_text=ytitle, # Y axisTitle
                         range=[start, end],
                         zeroline=True,
                         showline=True,  # Show y-axis line
                         linecolor='gray',  # Color of the y-axis line
                         titlefont_size=18,  # Set the title font size (optional)
                        titlefont_color="#008080",  # Set the title font color (optional)         )
                        )
    else:
        fig.update_xaxes(title_text=xtitle, # Y axisTitle
                         showline=True,  # Show y-axis line
                        linecolor='gray',
                        titlefont_size=18,  # Set the title font size (optional)
                        titlefont_color="#008080",  # Set the title font color (optional)
                         )
        
        fig.update_yaxes(title_text=ytitle, # Y axisTitle
                         showline=True,  # Show y-axis line
                         linecolor='gray',  # Color of the y-axis line
                         titlefont_size=18,  # Set the title font size (optional)
                        titlefont_color="#008080",  # Set the title font color (optional)         )
        )
        #  Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


#==========================================================================#
#                          Data Cleaning 1 Function
#==========================================================================#
# function to clean data 

target_mapping = {
    'C':0,
    'CL':1,
    'D':2
}

target_reverse_mapping = {
    0:'C',
    1:'CL',
    2:'D'
}


@st.cache_data()
def wrangle(df):
    # Add new Age column in years
    df["Age_In_Years"] = df['Age'] / 365.25
    
    # Change data types to boolean for better ML handling
    df["Ascites"] = df["Ascites"].apply(lambda x: 0 if x == "N" else 1).astype("int")
    df["Hepatomegaly"] = df["Hepatomegaly"].apply(lambda x: 0 if x == "N" else 1).astype("int")
    df["Spiders"] = df["Spiders"].apply(lambda x: 0 if x == "N" else 1).astype("int")
    df["Edema"] = df["Edema"].apply(lambda x: 0 if x == "N" else 1).astype("int")
    df["Drug"] = df["Drug"].apply(lambda x: 0 if x == "Placebo" else 1).astype("int")
    df["Sex"] = df["Sex"].apply(lambda x: 0 if x == "F" else 1).astype("int")
    
    
    # Drop N_Days to avoid Data leakage
    if 'N_Days' in df.columns:
        df.drop(['N_Days'],axis=1,inplace=True)
    if 'Status' in df.columns:
        df.Status = df.Status.map(target_mapping)
        df.Status = df.Status.astype(int)
    return df


#==========================================================================#
#                          Read Data
#==========================================================================#


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_clean = wrangle(df)
df_test_clean = wrangle(df_test)








#==========================================================================#
#                          ML Wrangle Function
#==========================================================================#


def ml_wrangle(df):
    
    # 4.1.1 Flag for normal Bilirubin levels Bilirubin_Normal column
    df["Bilirubin_Normal"] = df["Bilirubin"].apply(lambda x: 0 if x < 0.1 or x > 1.2 else 1).astype("int")

    #4.1.2 Flag for normal Cholesterol levels Cholesterol_Normal column¶
    df["Cholesterol_Normal"] = df["Cholesterol"].apply(lambda x: 0 if x > 200 else 1).astype("int")
    
    #4.1.3 Flag for normal Albumin levels Albumin_Normal column¶
    df["Albumin_Normal"] = df["Albumin"].apply(lambda x: 0 if x < 3.5 or x > 5 else 1).astype("int")

    #4.1.4 Flag for normal Copper levels Copper_Normal column
    df["Copper_Normal"] = df["Copper"].apply(lambda x: 0 if x < 70 or x > 150 else 1).astype("int")

    #4.1.5 Flag for normal Alk_Phos levels Alk_Phos_Normal column
    df["Alk_Phos_Normal"] = df["Alk_Phos"].apply(lambda x: 0 if x < 40 or x > 150 else 1).astype("int")

    #4.1.6 Flag for normal SGOT levels SGOT_Normal column¶
    df["SGOT_Normal"] = df["SGOT"].apply(lambda x: 0 if x < 8 or x > 40 else True).astype("int")

    #4.1.7 Flag for normal Tryglicerides levels Tryglicerides_Normal column¶
    df["Tryglicerides_Normal"] = df["Tryglicerides"].apply(lambda x: 0 if  x > 150 else 1).astype("int")

    #4.1.8 Flag for normal Platelets levels Platelets_Normal column
    df["Platelets_Normal"] = df["Platelets"].apply(lambda x: 0 if x < 150 or x > 450 else 1).astype("int")

    #4.1.9 Flag for normal Prothrombin levels Prothrombin_Normal column¶
    df["Prothrombin_Normal"] = df["Prothrombin"].apply(lambda x: 0 if x < 9.5 or x > 13.5 else 1).astype("int")

    # Mean for metrics
    df["Mean_Metrics"] = df[["Ascites", "Hepatomegaly", "Spiders", "Edema"]].mean(axis=1)

    #AST_ALT_Ratio
    
    df['AST_ALT_Ratio'] = df['SGOT'] / df['Alk_Phos']

    # Albumin to Bilirubin Ratio
    df['Albumin_Bilirubin_Ratio'] = df['Albumin'] / df['Bilirubin']
    # Interaction Terms
    df['Age_Bilirubin'] = df['Age_In_Years'] * df['Bilirubin']
    df['Albumin_Platelets'] = df['Albumin'] * df['Platelets']

    # pugh score 
    df['Child_Pugh_Proxy'] = (df['Bilirubin'] > 2).astype(int) + (df['Albumin'] < 2.8).astype(int) + df['Ascites'] + df['Hepatomegaly']


    

    for col in ['Bilirubin', 'Cholesterol', 'SGOT', 'Tryglicerides']:
        df[col] = np.log1p(df[col])
        df.rename(columns={col: col+"_log"}, inplace=True)
        
    return df




df_ml = ml_wrangle(df_clean)
df_test_ml = ml_wrangle(df_test_clean)




#==========================================================================#
#                          ML Models
#==========================================================================#




columns = ["Drug","Age_In_Years","Sex","Ascites","Hepatomegaly","Spiders","Edema","Bilirubin_log","Cholesterol_log","Albumin","Copper","Alk_Phos","SGOT_log","Tryglicerides_log","Platelets","Prothrombin","Stage","Mean_Metrics","AST_ALT_Ratio","Albumin_Bilirubin_Ratio","Age_Bilirubin","Albumin_Platelets","Child_Pugh_Proxy"]


#==========================================================================#
#                          ML Models
#==========================================================================#

X = df_ml[columns].copy()
y = df_ml.Status.copy()
test = df_test_ml[columns].copy()

# Split data 
random_state = 42
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=random_state)

# Proportions of all classes
class_proportions = {0: 0.628083, 2: 0.337128, 1: 0.034788}

# Calculate class weights as inverse of proportions
class_weights = {cls: 1.0 / proportion for cls, proportion in class_proportions.items()}



#==========================================================================#
#                         Light GBM
#==========================================================================#
@st.experimental_fragment()
@st.cache_data()
def ml_model(model_name):
    if model_name == "LighGBM":
        start_time = time.time()
        best_params = {"bagging_fraction":0.8147515753137136,"feature_fraction":0.7146909188772861,"min_child_samples":94,"num_leaves":92,"bagging_freq":3,"learning_rate":0.04702757643730806}
        model = LGBMClassifier(**best_params, random_state=random_state, class_weight=class_weights)
        

    elif model_name == "xgb":
        start_time = time.time()
        best_params = {'max_depth': 5, 'learning_rate': 0.08834858264808022, 'colsample_bytree': 0.7000005136716684, 'subsample': 0.8764008144523939, 'min_child_weight': 6, 'gamma': 0.00018161095420293524, 'lambda': 6.098912138616464e-07, 'alpha': 8.653528554492958e-05}
        model = xgb.XGBClassifier(**best_params, random_state=random_state, class_weight=class_weights)

    elif model_name == "rf":
        start_time = time.time()
        best_params = {'n_estimators': 718, 'max_depth': 24, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_features': 0.2800980471014083, 'bootstrap': False, 'criterion': 'entropy'}
        model = RandomForestClassifier(**best_params, random_state=random_state, class_weight=class_weights)
        


    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    # Predict on train and test sets
    train_preds_proba = model.predict_proba(train_X)
    test_preds_proba = model.predict_proba(test_X)

    # Calculate log loss on train and test sets
    train_log_loss = log_loss(train_y, train_preds_proba)
    test_log_loss = log_loss(test_y, test_preds_proba)


    # Evaluate model
    train_acc = model.score(train_X, train_y)
    test_acc = model.score(test_X, test_y)

    precision = precision_score(test_y, preds, average='weighted')
    recall = recall_score(test_y, preds, average='weighted')
    f1 = f1_score(test_y, preds, average='weighted')

    train_test_scores = {
    "Metric": [
        "LogLoss", 
        "Accuracy", 
    ],
    "Train_Score": [
        train_log_loss, 
        train_acc,
    ]
    ,
    "Test_Score": [
        test_log_loss,
        test_acc
    ]
}
    train_test_scores_df = pd.DataFrame(train_test_scores)

    all_scores = {
        "Metric": [
            "LogLoss", 
            "Accuracy", 
            "Precision", 
            "Recall", 
            " F1", 
        ],
        "Score": [
            round(train_log_loss,4), 
            round(train_acc,4), 
            round(precision,4), 
            round(recall,4), 
            round(f1,4)
        ],
        "Interpretation":[
            "This measures how well the model's predicted probabilities match the actual outcomes. A lower value indicates better performance." ,

            f"Accuracy: This tells us that the model correctly predicted the outcome about {round(train_acc, 2)*100}% of the time.",

            f"Precision: This indicates that when the model predicted a certain class, it was correct about {round(precision, 2)*100}% of the time.",

            f"Recall: This measures the model's ability to correctly identify all the relevant instances. The model successfully identified about {round(recall, 2)*100}% of the actual instances of a class.",

            f"F1 Score: This is a balance between precision and recall."]
    }
    all_scores_df = pd.DataFrame(all_scores)

    # Record the end time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time
    

    return all_scores_df, train_test_scores_df, model, execution_time
    


