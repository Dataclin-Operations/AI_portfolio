#==========================================================================#
#                          Import Libiraries
#==========================================================================#
import sys

import streamlit as st
import os

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency



st.set_page_config(page_title="Dashboard",
                #    page_icon = "⚕:",
                   layout="wide"
                   )





#==========================================================================#
#                         Import functions
#==========================================================================#


sys.path.append(os.path.abspath('./functions.py'))

from functions import bar_chart,  df_test_clean,df_clean






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


#==========================================================================#
#                          Pie charts Functions
#==========================================================================#


@st.experimental_fragment()
@st.cache_data()
def make_a_pie(data, names_,height_ , title_, legend, variable_name=False):
    if variable_name == False:
        colors = ['#bdbdbd' ,'#008294','#4b4b4c' ]
    elif variable_name == None:
        colors = ['#008294','#bdbdbd' ,'#4b4b4c' ]
    else:
        colors = ['#008294', '#4b4b4c' ,'#bdbdbd']

    fig = px.pie(data, values='proportion', names=names_,
                custom_data=['Count'],
                labels={'proportion': 'Percentage'},
                color_discrete_sequence=colors,
                height=height_)

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
        title={'text': title_, 'y': 0.95, 'x': 0.40, 'xanchor': 'center', 'yanchor': 'top'},  # Center and adjust title
        title_font_size=20,  # Title font size
        legend_title_text=legend,  # Legend title
        font=dict(family='Arial', size=14),  # Font family and size for labels
    )

    st.plotly_chart(fig)

# status pie
def status_pie():
    status_counts = df_clean['Status'].value_counts(normalize=True).to_frame().reset_index()
    status_counts["Count"] = df_clean['Status'].value_counts().values
    status_counts["proportion"] =  round(status_counts["proportion"]*100, 2)


    make_a_pie(status_counts, ["C (censored)", "D (deceased)", "CL (alive due to liver transplant)"],450 , "Patients Outcomes Proportions", "Patient Outcomes", True)

    


# Drug pie
def Drug_pie():
    Drug = df_clean['Drug'].value_counts(normalize=True).to_frame().reset_index()
    Drug["Count"] = df_clean['Drug'].value_counts().values
    Drug["proportion"] =  round(Drug["proportion"]*100, 2)

    make_a_pie(Drug, ["Placebo", "D-penicillamine"],450 , "Drug Type Proportion", "Drug Type", True)
   

     



# gender_pie
def gender_pie():
    gender = df_clean['Sex'].value_counts(normalize=True).to_frame().reset_index()
    gender["Count"] = df_clean['Sex'].value_counts().values
    gender["proportion"] =  round(gender["proportion"]*100, 2)
    gender["Sex"] = gender["Sex"].apply(lambda x: "F" if x ==0  else "M")

    make_a_pie(gender, ["Female", "Male"],400 , "Gender", "Patient Gender")

# ascites_pie
def ascites_pie():
    
    Ascites = df_clean['Ascites'].value_counts(normalize=True).to_frame().reset_index()
    Ascites["Count"] = df_clean['Ascites'].value_counts().values
    Ascites["proportion"] =  round(Ascites["proportion"]*100, 2)
    Ascites["Ascites"] = Ascites["Ascites"].apply(lambda x: "No" if x ==0  else "Yes")
    
    make_a_pie(Ascites, [i for i in Ascites.Ascites.values],400 , "Ascites Status", "Ascites")

# hepatomegaly_pie
def hepatomegaly_pie():
    Hepatomegaly = df_clean['Hepatomegaly'].value_counts(normalize=True).to_frame().reset_index()
    Hepatomegaly["Count"] = df_clean['Hepatomegaly'].value_counts().values
    Hepatomegaly["proportion"] =  round(Hepatomegaly["proportion"]*100, 2)
    Hepatomegaly["Hepatomegaly"] = Hepatomegaly["Hepatomegaly"].apply(lambda x: "No" if x ==0  else "Yes")
    make_a_pie(Hepatomegaly, [i for i in Hepatomegaly.Hepatomegaly.values],400 , "Hepatomegaly Status", "Hepatomegaly", None)
    

# spiders_pie
def spiders_pie():
    
    Spiders = df_clean['Spiders'].value_counts(normalize=True).to_frame().reset_index()
    Spiders["Count"] = df_clean['Spiders'].value_counts().values
    Spiders["proportion"] =  round(Spiders["proportion"]*100, 2)
    Spiders["Spiders"] = Spiders["Spiders"].apply(lambda x: "No" if x ==0  else "Yes")

    make_a_pie(Spiders, [i for i in Spiders.Spiders.values],400 , "Spiders Status", "Spiders")

# edema_pie
def edema_pie():
    Edema = df_clean['Edema'].value_counts(normalize=True).to_frame().reset_index()
    Edema["Count"] = df_clean['Edema'].value_counts().values
    Edema["proportion"] =  round(Edema["proportion"]*100, 2)
    Edema["Edema"] = Edema["Edema"].apply(lambda x: "No" if x ==0  else "Yes")

    make_a_pie(Edema,  [i for i in Edema.Edema.values],400 , "Edema Status", "Edema")
   
def stage_bar():
    Stage = df_clean['Stage'].value_counts(normalize=True).to_frame().reset_index()
    Stage["Count"] = df_clean['Stage'].value_counts().values
    Stage["proportion"] =  round(Stage["proportion"]*100, 1)
    Stage["proportions"] = Stage["proportion"].apply(lambda x: str(x)+" %")
    Stage["Stage"]= Stage["Stage"].map({3.0:"Stage 3", 4.0:"Stage 4", 2.0:"Stage 2", 1.0:"Stage 1"})
    Stage = Stage.sort_values(by="Stage")

    bar_chart(
    Stage,
    "Stage",
    "proportion",
    txt = "proportions",
    # labls = {'currentSmoker': 'Current Smoker', 'diabetes': 'Diabetes', 'per': 'CHD Risk', "annotation_per": "False"},
    hover={"proportion":False},
    colors =[ "#c6c6c6",'#4D4D4D','#008294',"#003d46","#001b1e"] ,
    hight=600,
    wdth=1100,
    ttle="Cirrhosis Stages",
    xtitle="Stage" ,
    ytitle="Stage Propotion %" ,
    colour='Stage', 
    bg=0.001, 
    bgg=0.001,
    yscale = False,
    yscale_percentage=False
)







#==========================================================================#
#                          Density charts Functions
#==========================================================================#




dist_columns = ["Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]

@st.experimental_fragment()
@st.cache_data()
def distribution_status(title):
    sns.set_style("whitegrid")
    df_den = df_clean.copy()
    df_den["Status"] = df_den["Status"].map(target_reverse_mapping)
    fig,ax = plt.subplots(5,2,figsize=(12,15))
    k=0
    j=0
    for col in dist_columns:
        sns.kdeplot(data = df_den, x =col, ax=ax[k,j],
                    shade=False,
                    hue="Status",
                    color='#003d46', 
                    linewidth=1, alpha=1,
                    palette=[ 'orange', 'firebrick', '#1c96c5'],
                    zorder=3,
                )
    
        ax[k,j].set_xlabel(col, fontsize=10, color="#008080")
        ax[k,j].set_ylabel("Density", fontsize=10, color="#008080")
        if j>=1:
            k+=1
            j=-1
        j+=1
    # plt.tight_layout()
    plt.subplots_adjust(hspace = 0.6, wspace=0.4)
    st.markdown("<h2 style='color: #008080; text-align:center'>Liver Function Biomarkers Density Stratified by Outcome</h2>", unsafe_allow_html=True)
    # fig.suptitle('Liver Function Biomarkers Density Stratified by '+ title, fontsize=20, color="k")
    ax[4,1].set_visible(False)

    # Display the plot in Streamlit
    st.pyplot(fig)



@st.experimental_fragment()
@st.cache_data()   
def distribution_drug(title):
    sns.set_style("whitegrid")
    df_den = df_clean.copy()
    df_den.Drug = df_den.Drug.apply(lambda x: "Placebo" if x == 0 else "D-penicillamine")

    fig,ax = plt.subplots(5,2,figsize=(12,15))
    k=0
    j=0
    for col in dist_columns:
        sns.kdeplot(data = df_den, x =col, ax=ax[k,j],
                    shade=False,
                    hue="Drug",
                    color='#003d46', 
                    linewidth=1, alpha=1,
                    zorder=3,
                )
    
        ax[k,j].set_xlabel(col, fontsize=10, color="#008080")
        ax[k,j].set_ylabel("Density", fontsize=10, color="#008080")
        if j>=1:
            k+=1
            j=-1
        j+=1
    # plt.tight_layout()
    plt.subplots_adjust(hspace = 0.6, wspace=0.4)
    st.markdown("<h2 style='color: #008080; text-align:center'>Liver Function Biomarkers Density Stratified by Drug</h2>", unsafe_allow_html=True)

    # fig.suptitle('Liver Function Biomarkers Density Stratified by '+ title, fontsize=20, color="k")
    ax[4,1].set_visible(False)

    # Display the plot in Streamlit
    st.pyplot(fig)



#==========================================================================#
#                          Distribtion charts Functions
#==========================================================================#

@st.experimental_fragment()
@st.cache_data()     
def vilion_drug(title):
    sns.set_style("whitegrid")
    df_den = df_clean.copy()
    # st.write(df_den.Drug.value_counts())
    df_den.Drug = df_den.Drug.apply(lambda x: "Placebo" if x == 0 else "D-penicillamine")
    fig,ax = plt.subplots(5,2,figsize=(12,15))
    k=0
    j=0
    for col in dist_columns:
        sns.violinplot(data = df_den, x="Drug", y=col, ax=ax[k, j],  palette=['#4D4D4D', '#008294', '#bdbdbd'], box=True )

    
        ax[k,j].set_xlabel(col, fontsize=10, color="#008080")
        ax[k,j].set_ylabel("Distribution", fontsize=10, color="#008080")
        if j>=1:
            k+=1
            j=-1
        j+=1
    # plt.tight_layout()
    plt.subplots_adjust(hspace = 0.6, wspace=0.4)
    st.markdown("<h2 style='color: #008080; text-align:center'>Liver Function Biomarkers Distribution Stratified by Drug</h2>", unsafe_allow_html=True)

    # fig.suptitle('Liver Function Biomarkers Distribution Stratified by '+ title, fontsize=20, color="k")
    ax[4,1].set_visible(False)

    # Display the plot in Streamlit
    st.pyplot(fig)


@st.experimental_fragment()
@st.cache_data()    
def vilion_outcome(title):
    sns.set_style("whitegrid")
    df_den = df_clean.copy()
    df_den.Status = df_den.Status.map(target_reverse_mapping)
    fig,ax = plt.subplots(5,2,figsize=(12,15))
    k=0
    j=0
    for col in dist_columns:
        sns.violinplot(data = df_den, x="Status", y=col, ax=ax[k, j],  palette=['#4D4D4D', '#008294', '#bdbdbd'], box=True )
    
        ax[k,j].set_xlabel(col, fontsize=10, color="#008080")
        ax[k,j].set_ylabel("Distribution", fontsize=10, color="#008080")
        if j>=1:
            k+=1
            j=-1
        j+=1
    # plt.tight_layout()
    plt.subplots_adjust(hspace = 0.6, wspace=0.4)
    st.markdown("<h2 style='color: #008080; text-align:center'>Liver Function Biomarkers Distribution Stratified by Outcome</h2>", unsafe_allow_html=True)
    # fig.suptitle('Liver Function Biomarkers Distribution Stratified by '+ title, fontsize=20, color="k")
    ax[4,1].set_visible(False)

    # Display the plot in Streamlit
    st.pyplot(fig)



#==========================================================================#
#                          Correlations charts Functions
#==========================================================================#




continuous_columns = ["Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]

@st.experimental_fragment()
@st.cache_data()    
def corr1():
    corr_data = df_clean[continuous_columns + ["Status"]].copy()


    def anova_test(df, continuous_col, categorical_col):
        model = ols(f'{continuous_col} ~ C({categorical_col})', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        eta_squared = anova_table['sum_sq'].iloc[0] / anova_table['sum_sq'].sum()
        p_value = anova_table['PR(>F)'].iloc[0]
        return eta_squared, p_value

    results = []

    for col in continuous_columns:
        eta_sq, anova_p = anova_test(corr_data, col, 'Status')
        results.append({
            'Column': col,
            'ANOVA Eta Squared': eta_sq,
            'ANOVA p-value': anova_p,
        })

    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    results_df['ANOVA Eta Squared'] = round(results_df['ANOVA Eta Squared'], 2)
    results_df['ANOVA p-value'] = round(results_df['ANOVA p-value'], 4)
    fig = px.bar(results_df, x='Column', y='ANOVA Eta Squared',text='ANOVA Eta Squared', color='Column',
                hover_data={'ANOVA Eta Squared':True,'ANOVA p-value':True},
                color_discrete_sequence=[
            '#003f5c', '#2f4b7c', '#665191', '#008294', 
            '#4D4D4D', '#bdbdbd', '#ff7c43', '#ffa600', '#ffd700'
        ],
                height=600, width=1400)


    fig.update_traces(textfont_size=15, textposition='outside')



    fig.update_layout(
        legend_title_text=None,
        #background 
        plot_bgcolor="white", bargap=0.5, title={'text': 'Correlation between Status outcomes and Continuous Variables ', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, #title center and size
        title_font_size=25,  # Title font size
        xaxis=dict(
            title='X Axis Title',  # Example X axis title
            titlefont_size=14,  # Size of the X axis title font
            tickfont_size=12,   # Size of the X axis tick labels font
            showline=True,  # Show x-axis line
            showgrid=False, # showgrid
            gridcolor="lightgray",
            titlefont_color="#008080"),
        yaxis=dict(
            title='Y Axis Title',  # Example Y axis title
            titlefont_size=14,  # Size of the Y axis title font
            tickfont_size=12,   # Size of the Y axis tick labels font
            range=[-1, 1], 
        
            showgrid=True,
             titlefont_color="#008080" # showgrid   
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
                    titlefont_size=18,  # Set the title font size (optional)
                    titlefont_color="#008080",  # Set the title font color (optional)
                    )

    fig.update_yaxes(title_text="Correlation", # Y axisTitle
                    range=[-1, 1],
                    showline=True,  # Show y-axis line
                    linecolor='gray',  # Color of the y-axis line
                    titlefont_size=18,  # Set the title font size (optional)
                    titlefont_color="#008080",  # Set the title font color (optional)         )
                    )
    fig.add_hline(y=0.6, line=dict(color="red", width=0.5, dash="dash"))
    fig.add_annotation(y=0.6, text="Strong positive correlation line", showarrow=False,
                        xanchor='right', yanchor='bottom', font=dict(color="red"))

    fig.add_hline(y=-0.6, line=dict(color="blue", width=0.5, dash="dash"))
    fig.add_annotation(y=-0.6, text="Strong negative correlation line", showarrow=False,
                        xanchor='right', yanchor='bottom', font=dict(color="blue"))

    st.plotly_chart(fig)


@st.experimental_fragment()
@st.cache_data()    
def corr2():
    def cramers_v_and_p(contingency_table):
        chi2, p, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))  # correction for bias
        r_corr = r - ((r-1)**2)/(n-1)
        k_corr = k - ((k-1)**2)/(n-1)
        cramers_v_value = np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))
        return cramers_v_value, p

    # Calculate Cramér's V and p-value for each binary column against 'status'
    binary_columns = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    results = []

    for col in binary_columns:
        contingency_table = pd.crosstab(df_clean[col], df_clean['Status'])
        cramers_v_value, p_value = cramers_v_and_p(contingency_table)
        results.append({'Column': col, 'Cramér\'s V': cramers_v_value, 'p-value': p_value})

    # Create DataFrame with results
    results_df2 = pd.DataFrame(results)
    results_df2["Cramér's V"] = round(results_df2["Cramér's V"] , 2)
    results_df2["p-value"] = round(results_df2["p-value"] , 2)
    # Print results
    # results_df2.style

    fig = px.bar(results_df2, x='Column', y="Cramér's V",text="Cramér's V", color='Column',
             hover_data={"Cramér's V":True,'p-value':True},
             color_discrete_sequence=['#008294', 
        '#4D4D4D', '#bdbdbd', '#ff7c43'
    ],

             height=600, width=1400)



    fig.update_traces(textfont_size=15, textposition='outside')



    fig.update_layout(
        legend_title_text=None,
        #background 
        plot_bgcolor="white", bargap=0.5, title={'text': 'Correlation between Status outcomes and Boolean variables ', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, #title center and size
        title_font_size=25,  # Title font size
        xaxis=dict(
            title='X Axis Title',  # Example X axis title
            titlefont_size=14,  # Size of the X axis title font
            tickfont_size=12,   # Size of the X axis tick labels font
            showline=True,  # Show x-axis line
            showgrid=False, # showgrid
            gridcolor="lightgray",
            titlefont_color="#008080"),
        yaxis=dict(
            title='Y Axis Title',  # Example Y axis title
            titlefont_size=14,  # Size of the Y axis title font
            tickfont_size=12,   # Size of the Y axis tick labels font
            range=[-1, 1],
            titlefont_color="#008080", 
        
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
                    titlefont_size=18,  # Set the title font size (optional)
                    titlefont_color="#008080",  # Set the title font color (optional)
                    )

    fig.update_yaxes(title_text="Correlation", # Y axisTitle
                    range=[-1, 1],
                    showline=True,  # Show y-axis line
                    linecolor='gray',  # Color of the y-axis line
                    titlefont_size=18,  # Set the title font size (optional)
                    titlefont_color="#008080",  # Set the title font color (optional)         )
                    )
    fig.add_hline(y=0.6, line=dict(color="red", width=0.5, dash="dash"))
    fig.add_annotation(y=0.6, text="Strong positive correlation line", showarrow=False,
                        xanchor='right', yanchor='bottom', font=dict(color="red"))

    fig.add_hline(y=-0.6, line=dict(color="blue", width=0.5, dash="dash"))
    fig.add_annotation(y=-0.6, text="Strong negative correlation line", showarrow=False,
                        xanchor='right', yanchor='bottom', font=dict(color="blue"))

    st.plotly_chart(fig)


def corr3():
    # Check correlation with heatmap

    fig, ax = plt.subplots(figsize=(15,3))
    corr_matrix = df_clean[continuous_columns].corr()
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_matrix, annot=True, mask=mask);

    plt.title( 'Continuous Variables Correlations Heatmap', fontsize=12, fontweight='bold', fontfamily='serif', pad=20);
    st.pyplot(fig)




#==========================================================================#
#                          Dashboard1 Functions
#==========================================================================#

#  1. What proportion of patients exhibits a ten-year risk of coronary heart disease (TenYearCHD)? (Pie chart)
def proptions():
   

    col1, col2 = st.columns(2)
    with col1:
        status_pie()
    with col2:
        Drug_pie()

    st.markdown("<h2 style='color: #008080; text-align:center'>Patients Biomarkers Proportions</h2>", unsafe_allow_html=True)
#################################################################################
    col1, col2 = st.columns(2)

    with col1:
        gender_pie()
        hepatomegaly_pie()
    with col2:
        ascites_pie()
        spiders_pie()

    col1, col2, col3 = st.columns(3)
    with col2:
        edema_pie()
    stage_bar()

    


def density():
    pass

def correlations():
    pass

if __name__ == "__main__":
    logo_path = "./TheLogo2.png"  # Replace with the path to your logo image
    st.sidebar.image(logo_path, use_column_width=True)
    st.sidebar.markdown("<h2 style='color: #008080; text-align:center'>Select Chart option</h2>", unsafe_allow_html=True)
    chart_type_select = st.sidebar.selectbox(" ", ["Proportions", "Distribution", "Density", "Correlations "], label_visibility="collapsed")
    
    if chart_type_select == "Proportions":
        proptions()
    elif chart_type_select == "Distribution":
        # st.sidebar.markdown("<h4 style='color: #008080; text-align:center'>Stratify by</h4>", unsafe_allow_html=True)
        selection2 = st.sidebar.radio("Stratify by", options=[ "Outcome", "Drug"])
        if selection2 == "Drug":
            vilion_drug("Drug")
        else:
            vilion_outcome("Outcome")

        
        
        density()
    elif chart_type_select == "Density":
        # st.sidebar.markdown("<h4 style='color: #008080; text-align:center'>Stratify by</h4>", unsafe_allow_html=True)
        selection = st.sidebar.radio("Stratify by", options=[ "Outcome", "Drug"])
        if selection == "Drug":
            distribution_drug("Drug")
        else:
            distribution_status("Outcome")

    else:
        corr1()
        corr2()
        corr3()
        
