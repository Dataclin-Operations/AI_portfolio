#==========================================================================#
#                          Import Libiraries
#==========================================================================#
import sys
# Caution: path[0] is reserved for script path (or '' in REPL)

import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from st_aggrid.shared import GridUpdateMode
from streamlit_extras.dataframe_explorer import dataframe_explorer 
import os





st.set_page_config(page_title="Home Page",
                   page_icon = "⚕:",
                   layout="wide"
                   )





#==========================================================================#
#                         Import functions
#==========================================================================#


sys.path.append(os.path.abspath('./functions.py'))


def make_dcs():
    data = {
        "Column": [
            "male", 
            "age", 
            "education", 
            "currentSmok", 
            "cigsPerDay", 
            "BPMeds", 
            "prevalentStroke", 
            "prevalentHyp", 
            "diabetes", 
            "totChol", 
            "sysBP", 
            "diaBP", 
            "BMI", 
            "heartRate", 
            "glucose", 
            "TenYearCHD"
        ],
        "Description": [
            "Patient Gender Binary (1 = Male, 0 = Female)", 
            "Patient age.", 
            "Patient education level.", 
            "Smoker or not Binary (1 = Yes, 0 = No)", 
            "Number of smoking cigars per day", 
            "Blood Pressure Medications.", 
            "Whether the individual has had a stroke previously Binary (1 = Yes, 0 = No)", 
            "Whether the individual has prevalent hypertension (high blood pressure) Binary (1 = Yes, 0 = No)", 
            "Whether the individual has diabetes. Binary (1 = Yes, 0 = No)", 
            "`Total Cholesterol` The total cholesterol level in the blood", 
            "`Systolic Blood Pressure` The pressure in the arteries when the heart beats", 
            "`Diastolic Blood Pressure` The pressure in the arteries when the heart is at rest between beats.", 
            "`Body Mass Index` A measure of body fat based on height and weight, calculated as weight (kg) / height (m)^2.", 
            "Heartbeats per minute.", 
            "The blood glucose level", 
            "`Ten-Year Coronary Heart Disease Risk` The estimated risk of developing CHD over the next ten years"
        ]
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # colored_header(
    # label="Dataset Description",
    # description="This is a description",
    # color_name="violet-70"
    # )

    st.header('Dataset Description', divider='blue')
    st.markdown("> #### The \"Framingham\" heart disease dataset includes over 4,240 records, 16 columns and 15 attributes. The goal of the dataset is to predict whether the patient has 10-year risk of future (CHD) coronary heart disease:")
    st.subheader('Columns Description', divider='blue')
    # st.write("Notebooks by notebook name:")

    gd = GridOptionsBuilder.from_dataframe(df)
    # gd.configure_pagination(enabled=True)
    # gd.configure_column("Upvotes", header_name="Upvotes",minWidth=50,groupable=True,aggFunc="sum")
    # gd.configure_column("URL", header_name="URL",minWidth=500,groupable=True,)
    # gd.configure_column("Notebook Title", header_name="Notebook Title",minWidth=400,groupable=True,)
    # gd.configure_column("Owner", header_name="Owner",minWidth=150,groupable=True,pivot=True, )
    gd.configure_default_column(
                    filter=True,autoSize=True,
                    resizable=True,
                    headerStyle={'textAlign': 'right', 'fontSize': '16px', 'fontWeight': 'bold', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': 'black'},  # Styling for header
                    cellStyle={'textAlign': 'left', 'fontSize': '16px', 'fontWeight': 'bold', 'fontFamily': 'Arial, sans-serif',}
                    )


    # gd.configure_side_bar()
    # gd.configure_selection(selection_mode="multiple",use_checkbox=True)
    gridoptions = gd.build()


    # Display the custom CSS
    grid_table = AgGrid(df.reset_index(),gridOptions=gridoptions,
                update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                height = 800,
                allow_unsafe_jscode=True,
                enable_enterprise_modules = True,
                theme = 'alpine',)




#==========================================================================#
#                          Pie charts Functions
#==========================================================================#


if __name__ == "__main__":
    
    st.image('./TheLogo2.png', use_column_width=True)        
    st.markdown("<h3 style='color: #008080; text-align:center'>Cirrhosis results from prolonged liver damage, leading to extensive scarring, often due to conditions like hepatitis or chronic alcohol consumption. The data provided is sourced from a Mayo Clinic study on primary biliary cirrhosis (PBC) of the liver carried out from 1974 to 1984.</h3>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: #008080; text-align:center'>Dataset Description</h1>", unsafe_allow_html=True)
    # st.markdown("<h3 style='color: #008080; text-align:left'>The dataset for this competition (both train and test) was generated from a deep learning model trained on the Cirrhosis Patient Survival Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.</h3>", unsafe_allow_html=True)
    st.markdown("""
| Column           | Description                                                                                                                                                                                 |
|:---------------: |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| **id**           | Unique identifier for each patient in the dataset.                                                                                                                                          |
| **N_Days**       | The number of days for which a patient was followed in the study.                                                                                                                           |
| **Drug**         | Indicates the medication a patient was prescribed in the study.<br> - D-penicillamine: A medication used for specific liver diseases.<br> - Placebo: A substance with no active ingredient used as a control.                     |
| **Age**          | The age of the patient in days.                                                                                                                                                             |
| **Sex**          | The patient's gender (male or female).                                                                                                                                                      |
| **Ascites**      | Refers to the presence or absence of fluid buildup in the abdomen, a potential complication of cirrhosis. (N = No, Y = Yes)                                                                 |
| **Hepatomegaly** | Whether the patient's liver was enlarged, another possible sign of cirrhosis. (N = No, Y = Yes)                                                                                             |
| **Spiders**      | Refers to the presence or absence of small, spider-like blood vessels on the skin, a sign sometimes seen in cirrhosis. (N = No, Y = Yes)                                                     |
| **Edema**        | Indicates swelling in the legs or feet, another potential complication of cirrhosis. (N = No, Y = Yes)                                                                                      |
| **Bilirubin**    | This is a substance produced by the liver. Abnormal levels can indicate liver problems.                                                                                                     |
| **Cholesterol**  | Cholesterol level in blood.                                                                                                                                                                 |
| **Albumin**      | Protein made by the liver. Low levels can be a sign of liver damage.                                                                                                                        |
| **Copper**       | Copper level in blood. High levels can be associated with liver problems.                                                                                                                   |
| **Alk_Phos**     | Alkaline Phosphatase, an enzyme found in the liver and other tissues. Abnormal levels can indicate liver damage.                                                                            |
| **SGOT**         | Serum Glutamic-Oxaloacetic Transaminase, an enzyme found in the liver and other tissues. Abnormal levels can indicate liver damage.                                                          |
| **Tryglicerides**| Type of fat found in the blood.                                                                                                                                                             |
| **Platelets**    | Cells in the blood that help with clotting.                                                                                                                                                 |
| **Prothrombin**  | Protein involved in blood clotting. Abnormal levels can indicate liver problems.                                                                                                            |
| **Stage**        | The severity or progression stage of the patient's cirrhosis.                                                                                                                               |
| **Status**       | Indicates the outcome for the patient at the end of the N_Days follow-up period. It can be one of three categories:<br> - **C (censored):** This means the patient was alive at the end of the N_Days period, but no information is available about their condition beyond that point.<br> - **CL (alive due to liver transplant):** This indicates the patient received a liver transplant during the N_Days period and was alive at the end of that timeframe.<br> - **D (deceased):** This signifies the patient passed away during the N_Days follow-up period. |
""", unsafe_allow_html=True)