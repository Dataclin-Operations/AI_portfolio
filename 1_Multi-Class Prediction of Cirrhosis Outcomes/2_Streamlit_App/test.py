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


#==========================================================================#
#                          Page Configs
#==========================================================================#


st.set_page_config(page_title="EDA Dashboard",
                   page_icon = "⚕:",
                   layout="wide"
                   )





#==========================================================================#
#                          Data Cleaning 2 Function
#==========================================================================#


sys.path.append(os.path.abspath('./functions.py'))

from functions import bar_chart, df_clean, df_test_clean



#==========================================================================#
#                          Dashboard1 Functions
#==========================================================================#

#  1. What proportion of patients exhibits a ten-year risk of coronary heart disease (TenYearCHD)? (Pie chart)
def proptions():
    status_counts = df_clean['Status'].value_counts(normalize=True).to_frame().reset_index()
    status_counts["Count"] = df_clean['Status'].value_counts().values
    status_counts["proportion"] =  round(status_counts["proportion"]*100, 2)
    status_counts


    fig = px.pie(status_counts, values='proportion', names=["C (censored)", "D (deceased)", "CL (alive due to liver transplant)"],
             custom_data=['Count'],
             labels={'proportion': 'Percentage'},
             color_discrete_sequence=['#008294', '#4b4b4c','#bdbdbd'],
             height=450)

    fig.update_traces(
        marker_line_color='black',  # Marker line color
        marker_line_width=0.5,  # Marker line width
        opacity=1,
        hoverinfo='label+value',  # Display label and value on hover
        texttemplate='%{value:.2f}',
        textinfo='percent', 
        textfont_size=16,
        hovertemplate='<b>%{label}</b><br>Count: %{customdata[0]}<extra></extra>'  # Custom hover template to show only count
    )

    fig.update_layout(
        title={'text': 'Proportion of Patients Outcomes', 'y': 0.95, 'x': 0.45, 'xanchor': 'center', 'yanchor': 'top'},  # Center and adjust title
        title_font_size=20,  # Title font size
        legend_title_text='Patient Outcomes',  # Legend title
        font=dict(family='Arial', size=14),  # Font family and size for labels
    )

    st.plotly_chart(fig)





def metrics(label, var):
    st.markdown(
                f"""
                <div style="
                    background-color: #f0f0f0;
                    padding: 10px;
                    border-radius: 15px;
                    border: 5px solid #4CAF50;
                    text-align: center;
                    font-size: 1.4em;
                    color: #4CAF50;
                    font-weight: bold;
                    margin:10px;
                ">
                    {label}: {var:,}
                </div>
                """,
                unsafe_allow_html=True
            )
    return var




#==========================================================================#
#                          2. Smoke
#==========================================================================#


def smoke_diabet():
    grouped_chd_dia = df_clean.groupby(['currentSmoker',"diabetes"]).TenYearCHD.mean().reset_index(name='per')
    grouped_chd_dia["currentSmoker"] = grouped_chd_dia["currentSmoker"].apply(lambda x : "Smoker" if x == 1 else "Non-Smoker")
    grouped_chd_dia["diabetes"] = grouped_chd_dia["diabetes"].apply(lambda x : "Diabetes" if x == 1 else "Non-Diabetes")
    # k["per"] = round(k["per"]*100, 2)
    grouped_chd_dia["annotation_per"] = round(grouped_chd_dia["per"]*100, 0)
    grouped_chd_dia["annotation_per"] = grouped_chd_dia["annotation_per"].astype(int)
    grouped_chd_dia["annotation_per"] = grouped_chd_dia["annotation_per"].apply(lambda x : str(x) + " %")


    bar_chart(
    grouped_chd_dia,
    "currentSmoker",
    "per",
    txt = "annotation_per",
    labls = {'currentSmoker': 'Current Smoker', 'diabetes': 'Diabetes', 'per': 'CHD Risk', "annotation_per": "False"},
    hover={"annotation_per":False},
    colors =['#4D4D4D','#008294'] ,
    wdth=900,
    ttle="10-Year CHD Risk Percentage: Smoking Status vs. Diabetes",
    xtitle="Smoking Status" ,
    ytitle="10-Year CHD Risk %" ,
    colour='diabetes', 
    bg=0.6, 
    bgg=0.1,
    yscale = False,
    yscale_percentage=True,
    hight=500
)
    

def smoke_stroke():
    grouped_chd_pstro = df_clean.groupby(['currentSmoker',"prevalentStroke"]).TenYearCHD.mean().reset_index(name='per')
    grouped_chd_pstro["currentSmoker"] = grouped_chd_pstro["currentSmoker"].apply(lambda x : "Smoker" if x == 1 else "Non-Smoker")
    grouped_chd_pstro["prevalentStroke"] = grouped_chd_pstro["prevalentStroke"].apply(lambda x : "Had Stroke" if x == 1 else "No-Stroke")
    # k["per"] = round(k["per"]*100, 2)
    grouped_chd_pstro["annotation_per"] = round(grouped_chd_pstro["per"]*100, 0)
    grouped_chd_pstro["annotation_per"] = grouped_chd_pstro["annotation_per"].astype(int)
    grouped_chd_pstro["annotation_per"] = grouped_chd_pstro["annotation_per"].apply(lambda x : str(x) + " %")

    bar_chart(
    grouped_chd_pstro,
    "currentSmoker",
    "per",
    txt = "annotation_per",
    labls = {'currentSmoker': 'Current Smoker', 'prevalentStroke': 'Prevalent Stroke', 'per': 'CHD Risk', "annotation_per": "False"},
    hover={"annotation_per":False},
    colors =['#8A8D90','#1192e9'] ,
    hight=500,
    wdth=900,
    ttle="10-Year CHD Risk Percentage: Smoking Status vs. Prevalent Stroke",
    xtitle="Smoking Status" ,
    ytitle="10-Year CHD Risk %" ,
    colour='prevalentStroke', 
    bg=0.6, 
    bgg=0.1,
    yscale = False,
    yscale_percentage=True
)



def smoke_med():
    grouped_chd_bp = df_clean.groupby(['currentSmoker',"BPMeds"]).TenYearCHD.mean().reset_index(name='per')
    grouped_chd_bp["currentSmoker"] = grouped_chd_bp["currentSmoker"].apply(lambda x : "Smoker" if x == 1 else "Non-Smoker")
    grouped_chd_bp["BPMeds"] = grouped_chd_bp["BPMeds"].astype(int)
    grouped_chd_bp["BPMeds"] = grouped_chd_bp["BPMeds"].astype(str)
    dic5 = {
        "0": "No Meds",
        "1": "Med Users",
        "-1": "NA",
        }
    grouped_chd_bp["BPMeds"] = grouped_chd_bp["BPMeds"].map(dic5)
    # k["per"] = round(k["per"]*100, 2)
    grouped_chd_bp["annotation_per"] = round(grouped_chd_bp["per"]*100, 0)
    grouped_chd_bp["annotation_per"] = grouped_chd_bp["annotation_per"].astype(int)
    grouped_chd_bp["annotation_per"] = grouped_chd_bp["annotation_per"].apply(lambda x : str(x) + " %")

    bar_chart(
    grouped_chd_bp,
    "currentSmoker",
    "per",
    txt = "annotation_per",
    labls = {'currentSmoker': 'Current Smoker', 'BPMeds': 'Blood Presure Medication', 'per': 'CHD Risk', "annotation_per": "False"},
    hover={"annotation_per":False},
    colors =["#1e3d59",'silver','#FF7F0E',] ,
    hight=500,
    wdth=900,
    ttle="10-Year CHD Risk Percentage: Smoking Status vs. Blood Presure Medication",
    xtitle="Smoking Status" ,
    ytitle="10-Year CHD Risk %" ,
    colour='BPMeds', 
    bg=0.5, 
    bgg=0.1,
    yscale = False,
    yscale_percentage=True,
    leg="Blood Presure Medication"
)




def smoke_hyper():
    grouped_chd_hyper_smoke = df_clean.groupby(['currentSmoker',"prevalentHyp"]).TenYearCHD.mean().reset_index(name='per')
    grouped_chd_hyper_smoke["currentSmoker"] = grouped_chd_hyper_smoke["currentSmoker"].apply(lambda x : "Smoker" if x == 1 else "Non-Smoker")
    grouped_chd_hyper_smoke["prevalentHyp"] = grouped_chd_hyper_smoke["prevalentHyp"].apply(lambda x : "Had Hypertension" if x == 1 else "No-Hypertension")

    grouped_chd_hyper_smoke["annotation_per"] = round(grouped_chd_hyper_smoke["per"]*100, 0)
    grouped_chd_hyper_smoke["annotation_per"] = grouped_chd_hyper_smoke["annotation_per"].astype(int)
    grouped_chd_hyper_smoke["annotation_per"] = grouped_chd_hyper_smoke["annotation_per"].apply(lambda x : str(x) + " %")


    bar_chart(
    grouped_chd_hyper_smoke,
    "currentSmoker",
    "per",
    txt = "annotation_per",
    labls = {'currentSmoker': 'Current Smoker', 'prevalentHyp': 'Prevalent Hypertension', 'per': 'CHD Risk', "annotation_per": "False"},
    hover={"annotation_per":False},
    colors =['#7a2048','#408ec6'] ,
    hight=500,
    wdth=900,
    ttle="10-Year CHD Risk Percentage: Smoking Status vs. Prevalent Hypertension",
    xtitle="Smoking Status" ,
    ytitle="10-Year CHD Risk %" ,
    colour='prevalentHyp', 
    bg=0.6, 
    bgg=0.1,
    yscale = False,
    yscale_percentage=True
)


#==========================================================================#
#                          3. Gender
#==========================================================================#


def gender_diabetic():
    grouped_chd_gender = df_clean.groupby(['male',"diabetes"]).TenYearCHD.mean().reset_index(name='per')
    grouped_chd_gender["male"] = grouped_chd_gender["male"].apply(lambda x : "Male" if x == 1 else "Female")
    grouped_chd_gender["diabetes"] = grouped_chd_gender["diabetes"].apply(lambda x : "Diabetes" if x == 1 else "Non-Diabetes")

    grouped_chd_gender["annotation_per"] = round(grouped_chd_gender["per"]*100, 0)
    grouped_chd_gender["annotation_per"] = grouped_chd_gender["annotation_per"].astype(int)
    grouped_chd_gender["annotation_per"] = grouped_chd_gender["annotation_per"].apply(lambda x : str(x) + " %")

    bar_chart(
    grouped_chd_gender,
    "male",
    "per",
    txt = "annotation_per",
    labls = {'male': 'Gender', 'diabetes': 'Diabetes', 'per': 'CHD Risk'},
    hover={"annotation_per":False},
    colors =['#4D4D4D','#008294'] ,
    hight=500,
    wdth=900,
    ttle="10-Year CHD Risk Percentage: Gender vs. Diabetes",
    xtitle="Gender" ,
    ytitle="10-Year CHD Risk %" ,
    colour='diabetes', 
    bg=0.6, 
    bgg=0.1,
    yscale = False,
    yscale_percentage=True
)

def gender_stroke():
    grouped_chd_pstro = df_clean.groupby(['male',"prevalentStroke"]).TenYearCHD.mean().reset_index(name='per')
    grouped_chd_pstro["male"] = grouped_chd_pstro["male"].apply(lambda x : "Male" if x == 1 else "Female")
    grouped_chd_pstro["prevalentStroke"] = grouped_chd_pstro["prevalentStroke"].apply(lambda x : "Had Stroke" if x == 1 else "No-Stroke")

    grouped_chd_pstro["annotation_per"] = round(grouped_chd_pstro["per"]*100, 0)
    grouped_chd_pstro["annotation_per"] = grouped_chd_pstro["annotation_per"].astype(int)
    grouped_chd_pstro["annotation_per"] = grouped_chd_pstro["annotation_per"].apply(lambda x : str(x) + " %")

    bar_chart(
    grouped_chd_pstro,
    "male",
    "per",
    txt = "annotation_per",
    labls = {'male': 'Gender', 'prevalentStroke': 'Prevalent Stroke', 'per': 'CHD Risk', "annotation_per": "False"},
    hover={"annotation_per":False},
    colors =['#8A8D90','#1192e9'] ,
    hight=500,
    wdth=900,
    ttle="10-Year CHD Risk Percentage: Gender vs. Prevalent Stroke",
    xtitle="Gender" ,
    ytitle="10-Year CHD Risk %" ,
    colour='prevalentStroke', 
    bg=0.6, 
    bgg=0.1,
    yscale = False,
    yscale_percentage=True
)




def gender_med():
    grouped_chd_med = df_clean.groupby(['male',"BPMeds"]).TenYearCHD.mean().reset_index(name='per')
    grouped_chd_med["male"] = grouped_chd_med["male"].apply(lambda x : "Male" if x == 1 else "Female")
    grouped_chd_med["BPMeds"] = grouped_chd_med["BPMeds"].astype(int)
    grouped_chd_med["BPMeds"] = grouped_chd_med["BPMeds"].astype(str)
    dic5 = {
        "0": "No Meds",
        "1": "Med Users",
        "-1": "NA",
        }
    grouped_chd_med["BPMeds"] = grouped_chd_med["BPMeds"].map(dic5)
    # k["per"] = round(k["per"]*100, 2)
    grouped_chd_med["annotation_per"] = round(grouped_chd_med["per"]*100, 0)
    grouped_chd_med["annotation_per"] = grouped_chd_med["annotation_per"].astype(int)
    grouped_chd_med["annotation_per"] = grouped_chd_med["annotation_per"].apply(lambda x : str(x) + " %")




    bar_chart(
    grouped_chd_med,
    "male",
    "per",
    txt = "annotation_per",
    labls = {'male': 'Gender', 'BPMeds': 'Blood Presure Medication', 'per': 'CHD Risk', "annotation_per": False},
    hover={"annotation_per":False},
    colors =["#1e3d59",'silver','#FF7F0E',] ,
    hight=500,
    wdth=900,
    ttle="10-Year CHD Risk Percentage: Gender vs. Blood Presure Medication",
    xtitle="Gender" ,
    ytitle="10-Year CHD Risk %" ,
    colour='BPMeds', 
    bg=0.5, 
    bgg=0.1,
    yscale = False,
    yscale_percentage=True,
    leg="Blood Presure Medication"
)


def gender_hyper():
    grouped_chd_hyper = df_clean.groupby(['male',"prevalentHyp"]).TenYearCHD.mean().reset_index(name='per')
    grouped_chd_hyper["male"] = grouped_chd_hyper["male"].apply(lambda x : "Male" if x == 1 else "Female")
    grouped_chd_hyper["prevalentHyp"] = grouped_chd_hyper["prevalentHyp"].apply(lambda x : "Had Hypertension" if x == 1 else "No-Hypertension")

    grouped_chd_hyper["annotation_per"] = round(grouped_chd_hyper["per"]*100, 0)
    grouped_chd_hyper["annotation_per"] = grouped_chd_hyper["annotation_per"].astype(int)
    grouped_chd_hyper["annotation_per"] = grouped_chd_hyper["annotation_per"].apply(lambda x : str(x) + " %")


    bar_chart(
    grouped_chd_hyper,
    "male",
    "per",
    txt = "annotation_per",
    labls = {'male': 'Gender', 'prevalentHyp': 'Prevalent Hypertension', 'per': 'CHD Risk', "annotation_per": "False"},
    hover={"annotation_per":False},
    colors =['#7a2048','#408ec6'] ,
    hight=500,
    wdth=900,
    ttle="10-Year CHD Risk Percentage: Gender vs. Prevalent Hypertension",
    xtitle="Gender" ,
    ytitle="10-Year CHD Risk %" ,
    colour='prevalentHyp', 
    bg=0.6, 
    bgg=0.1,
    yscale = False,
    yscale_percentage=True,
    leg="prevalentHyp"
)

    
#==========================================================================#
#                          4. Age
#==========================================================================#

def age_diabetic():
    grouped_chd_age = df_clean.groupby(['ageGroup',"diabetes"]).TenYearCHD.mean().reset_index(name='per')
    grouped_chd_age["diabetes"] = grouped_chd_age["diabetes"].apply(lambda x : "Diabetes" if x == 1 else "Non-Diabetes")

    grouped_chd_age["annotation_per"] = round(grouped_chd_age["per"]*100, 0)
    grouped_chd_age["annotation_per"] = grouped_chd_age["annotation_per"].astype(int)
    grouped_chd_age["annotation_per"] = grouped_chd_age["annotation_per"].apply(lambda x : str(x) + " %")
    grouped_chd_age.dropna(inplace=True)

    bar_chart(
    grouped_chd_age,
    "ageGroup",
    "per",
    txt = "annotation_per",
    labls = {'ageGroup': 'Age Group', 'diabetes': 'Diabetes', 'per': 'CHD Risk'},
    hover={"annotation_per":False},
    colors =['#bdbdbd','#008294'] ,
    hight=500,
    wdth=900,
    ttle="10-Year CHD Risk Percentage: Age Group vs. Diabetes",
    xtitle="Age Group" ,
    ytitle="10-Year CHD Risk %" ,
    colour='diabetes', 
    bg=0.6, 
    bgg=0.1,
    yscale = False,
    yscale_percentage=True
)


#==========================================================================#
#                          Correlations
#==========================================================================#
def correlations():
    dic = {"correlation":[], "p_value" :[]}
    columns = ["age", "cigsPerDay", "sysBP", "diaBP", "BMI"]
    for i in columns:
        r_pb, p_value = stats.pointbiserialr(df_clean[i], df_clean["TenYearCHD"])
        dic["correlation"].append(round(r_pb, 2))
        dic["p_value"].append(round(p_value, 4))
        
    corr_df = pd.DataFrame(dic)
    corr_df.index = ["Age", "Cigars per day", "Systolic BP","Diastolic BP", "BMI"]
    corr_df = corr_df.reset_index()
    corr_df = corr_df.rename(columns={"index": "Features"})

    fig = px.bar(corr_df, x='Features', y='correlation',text='correlation',
             labels={'TenYearCHD': '10-year CHD risk', 'count': 'Count', 'bmi_group_per': 'BMI Group'}, color='Features',
             hover_data={'correlation':True,'p_value':True},
             color_discrete_sequence = ['#008294', '#4D4D4D', '#bdbdbd', '#FF7F0E', '#f3ca20 ']

            # color_discrete_sequence=['#008294','#4D4D4D','#bdbdbd','#5A5A8B','#FF7F0E','#008294','#4D4D4D','#bdbdbd','#5A5A8B','#FF7F0E','#008080',]
             ,height=400, width=1100)



    fig.update_traces(textfont_size=15, textposition='outside')



    fig.update_layout(
        legend_title_text=None,
        #background 
         bargap=0.5, title={'text': 'Correlation between 10-year CHD and (Age, Cigars per day, Blood pressure and BMI) ', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, #title center and size
        title_font_size=20,  # Title font size
        xaxis=dict(
            title='X Axis Title',  # Example X axis title
            titlefont_size=14,  # Size of the X axis title font
            tickfont_size=12,   # Size of the X axis tick labels font
            showline=True,  # Show x-axis line
            showgrid=False, # showgrid
            gridcolor="lightgray",),
        yaxis=dict(
            title='Y Axis Title',  # Example Y axis title
            titlefont_size=14,  # Size of the Y axis title font
            tickfont_size=12,   # Size of the Y axis tick labels font
            range=[-1, 1], 
        
            showgrid=True, # showgrid   
        ),
        legend=dict(
            font_size=12,  # Size of the legend font
        ),

        # font
        font=dict(
            family='Arial', size=14  # Overall font size
        )
    )

    fig.update_xaxes(title_text="Features", # Y axisTitle
                    showline=True,  # Show y-axis line
                    linecolor='gray',
                    titlefont_size=14,  # Set the title font size (optional)
                    titlefont_color="white",  # Set the title font color (optional)
                    )

    fig.update_yaxes(title_text="Correlation", # Y axisTitle
                    range=[-1, 1],
                    showline=True,  # Show y-axis line
                    linecolor='gray',  # Color of the y-axis line
                    titlefont_size=16,  # Set the title font size (optional)
                    titlefont_color="#008080",  # Set the title font color (optional)         )
                    )
    fig.add_hline(y=0.7, line=dict(color="red", width=1, dash="dash"))
    fig.add_annotation(y=0.7, text="Strong positive correlation line", showarrow=False,
                        xanchor='center', yanchor='bottom', font=dict(color="red"))

    fig.add_hline(y=-0.7, line=dict(color="blue", width=1, dash="dash"))
    fig.add_annotation(y=-0.7, text="Strong negative correlation line", showarrow=False,
                        xanchor='center', yanchor='bottom', font=dict(color="blue"))
    # Show the plot
    st.plotly_chart(fig)

#==========================================================================#
#                          Dashboards
#==========================================================================#

def dashboard1():
    total_patients = df_clean.shape[0]
    chd_patients = df_clean[df_clean['TenYearCHD']== 1].shape[0]
    no_chd_patients = df_clean[df_clean['TenYearCHD']== 0].shape[0]

    proptions()
    col1, col2 = st.columns(2)
    with col1:
        correlations()
    with col2:
        col1, col2, col3 = st.columns(3)
        with col2:
            metrics("Total Patients",total_patients)
        col1, col2 = st.columns(2)
        # with col1:
        #     metrics("CHD Patients",chd_patients)
        # with col2:
        #     metrics("No_CHD Patients",no_chd_patients)
        chd_proptions()

    st.sidebar.header('Select option to stratify chart with', divider='blue')
    bar_selection = st.sidebar.selectbox(" ", ["Smoking Status", "Gender"], label_visibility="collapsed")
    # with col1:   
    #     st.subheader('Select option to stratify chart with', divider='blue')

    #     bar_selection = st.selectbox( "",["Smoking Status", "Gender"])
    if bar_selection == "Smoking Status":
        n_rows = 2
        rows = [st.container() for _ in range(n_rows)]
        cols_per_row = [r.columns(2) for r in rows]
        cols = [column for row in cols_per_row for column in row]

        with cols[0]:
            smoke_diabet()
        with cols[1]:
            smoke_hyper()
        with cols[2]:
            smoke_stroke()
        with cols[3]:
            smoke_med()
       

        # col1, col2 = st.columns(2)
        # with col1:
        #     smoke_diabet()
        #     smoke_stroke()
        # with col2:
        #     smoke_hyper()
        #     smoke_med()
            
    elif bar_selection == "Gender":
        col1, col2 = st.columns(2)
        with col1:
            gender_diabetic()
            gender_stroke()
        with col2:
            gender_hyper()
            gender_med()


#==========================================================================#
#                          Dashboard2 Functions
#==========================================================================#
def dashboard2():
    pass
    
    # for image_index, cat_image in enumerate(cat_images):
    #     cols[image_index].write(cat_image)

#==========================================================================#
#                          Dashboard3 Functions
#==========================================================================#


def dashboard3():
    pass






















#==========================================================================#
#                          Main Application
#==========================================================================#














if __name__ == "__main__":
    # if st.button("⚡"):
    #         st.experimental_rerun()
    
    logo_path = "./p.png"  # Replace with the path to your logo image
    st.sidebar.image(logo_path, use_column_width=True)
    dashboard1()