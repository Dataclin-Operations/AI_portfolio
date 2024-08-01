# #==========================================================================#
# #                          Import Libiraries
# #==========================================================================#
import sys
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






#==========================================================================#
#                          Import Data and functions From main file
#==========================================================================#

sys.path.append(os.path.abspath('../functions.py'))

from functions import df_clean_box, bar_chart, density





#==========================================================================#
#                          Cholesterol Functions
#==========================================================================#

def cholesterol_age():
    bar_chart(
    df_clean_box,
    "TenYearCHD",
    "totChol",
    colors =['#4D4D4D','#008294','#bdbdbd','#5A5A8B', "orange"],
    hight=500,
    wdth=900,
    ttle="Distribution of Cholesterol Levels impact for 10-Year CHD Risk, Varies by Age Group ",
    xtitle="Ten-Year CHD Risk" ,
    ytitle="Cholesterol leveL" ,
    colour='ageGroup', 
    bg=0.6, 
    bgg=0.1,
    leg="Age Group",
    box=True
)


def cholesterol_diab():
    bar_chart(
    df_clean_box,
    "TenYearCHD",
    "totChol",
    colors =['#4D4D4D','#008294','#bdbdbd','#008080','#5A5A8B'],
    hight=500,
    wdth=900,
    ttle="Distribution of Cholesterol Levels impact for 10-Year CHD Risk, Varies by Diabetes ",
    xtitle="Ten-Year CHD Risk" ,
    ytitle="Cholesterol leveL" ,
    colour='diabetes', 
    bg=0.6, 
    bgg=0.1,
    box=True
)


def cholesterol_stroke():
    bar_chart(
    df_clean_box,
    "TenYearCHD",
    "totChol",
    colors =['#4D4D4D','#008294','#bdbdbd','#008080','#5A5A8B'],
    hight=500,
    wdth=900,
    ttle="Distribution of Cholesterol Levels impact for 10-Year CHD Risk, Varies by Prevalent Stroke ",
    xtitle="Ten-Year CHD Risk" ,
    ytitle="Cholesterol leveL" ,
    colour='prevalentStroke', 
    bg=0.6, 
    bgg=0.8,
    leg="Prevalent Stroke",
    box=True
)






#==========================================================================#
#                          Heart Rate Functions
#==========================================================================#

def heart_age():
    bar_chart(
    df_clean_box,
    "TenYearCHD",
    "heartRate",
    colors =['#4D4D4D','#008294','#bdbdbd','#5A5A8B', "orange"],
    hight=500,
    wdth=900,
    ttle="Distribution of Heart Rate impact for 10-Year CHD Risk, Varies by Age Group ",
    xtitle="Ten-Year CHD Risk" ,
    ytitle="Heart Rate" ,
    colour='ageGroup', 
    bg=0.6, 
    bgg=0.1,
    leg="Age Group",
    box=True
)


def heart_diab():
    bar_chart(
    df_clean_box,
    "TenYearCHD",
    "heartRate",
    colors =['#4D4D4D','#008294','#bdbdbd','#008080','#5A5A8B'],
    hight=500,
    wdth=900,
    ttle="Distribution of Heart Rate impact for 10-Year CHD Risk, Varies by Diabetes ",
    xtitle="Ten-Year CHD Risk" ,
    ytitle="Heart Rate" ,
    colour='diabetes', 
    bg=0.6, 
    bgg=0.1,
    box=True
)



def heart_stroke():
    bar_chart(
    df_clean_box,
    "TenYearCHD",
    "heartRate",
    colors =['#4D4D4D','#008294','#bdbdbd','#008080','#5A5A8B'],
    hight=500,
    wdth=900,
    ttle="Distribution of Heart Rate impact for 10-Year CHD Risk, Varies by Prevalent Stroke ",
    xtitle="Ten-Year CHD Risk" ,
    ytitle="Heart Rate" ,
    colour='prevalentStroke', 
    bg=0.6, 
    bgg=0.1,
    leg="Prevalent Stroke",
    box=True
)





def dashboard1():
    st.sidebar.header('Select option to stratify chart with', divider='blue')
    bar_selection = st.sidebar.selectbox(" ", ["Heart Rate", "Prevalent Stroke"], label_visibility="collapsed")

    if bar_selection == "Heart Rate":
        heart_age()
        col1, col2= st.columns(2)
        with col1:
            heart_diab()
        with col2:
            heart_stroke()

    elif bar_selection == "Prevalent Stroke":
        cholesterol_age()
        col1, col2= st.columns(2)
        with col1:
           cholesterol_diab()
        with col2:
            cholesterol_stroke()
    
#==========================================================================#
#                          Main Application
#==========================================================================#


if __name__ == "__main__":
    # if st.button("⚡"):
    #         st.experimental_rerun()
    
    logo_path = "./p.png"  # Replace with the path to your logo image
    st.sidebar.image(logo_path, use_column_width=True)
    dashboard1()


   

