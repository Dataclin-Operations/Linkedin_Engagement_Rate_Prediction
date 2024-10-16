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

from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
from st_aggrid.shared import GridUpdateMode


import shap

import warnings
# Suppress specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)



#==========================================================================#
#                          Import Data and functions From main file
#==========================================================================#

sys.path.append(os.path.abspath('../functions.py'))

from functions import test_X, ml_model






@st.fragment()
def disply_shap_values(table, key):
    gd = GridOptionsBuilder.from_dataframe(table)
    gd.configure_column("ID", header_name="ID",minWidth=20,groupable=False,filter=True,autoSize=True,
                    resizable=True,
                    # headerStyle={'textAlign': 'center', 'fontSize': '50px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': '#008080'},  # Styling for header
                    cellStyle={'textAlign': 'left', 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif','color': '#008080'}
                    
                        )
    gd.configure_column("Feature", header_name="Feature",minWidth=500,groupable=False,filter=True,autoSize=True,
                    resizable=True,
                    # headerStyle={'textAlign': 'center', 'fontSize': '50px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f0f0', 'color': '#008080'},  # Styling for header
                    cellStyle={'textAlign': 'left', 'fontSize': '16px', 'fontFamily': 'Arial, sans-serif','color': '#008080'}
                    
                        )
    gd.configure_column("Mean(|SHAP VALUES|)", header_name="Mean(|SHAP VALUES|)",minWidth=150,groupable=True,filter=True,autoSize=False,
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
    
    grid_table = AgGrid(table.reset_index(),gridOptions=gridoptions,
                update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                height = 290,
                allow_unsafe_jscode=True,
                enable_enterprise_modules = True,
                theme = 'alpine',columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS, key=key)


#==========================================================================#
#                          Shap Values
#==========================================================================#
@st.fragment()
def shap_explainer_chart(_selected_model, chart_type):
    
    start_time = time.time()
    explainer = shap.TreeExplainer(selected_model)
    shap_values = explainer.shap_values(test_X)
    plt.figure(figsize=(3, 1))
    if chart_type =="Bar":
        # col1, col2, col3, col4, col5 = st.columns(5)
        # with col2:
        #     st.markdown("<h5 style='color: #008080; text-align:center'>Class Mapping</h5>", unsafe_allow_html=True)
        # with col3:
        #     st.markdown("<h5 style='color: #008080; text-align:center'>Class 0: <b style='color:red'>C</b></h5>", unsafe_allow_html=True)
        # with col4:
        #     st.markdown("<h5 style='color: #008080; text-align:center'>Class 1: <b style='color:red'>CL</b></h5>", unsafe_allow_html=True)
        # with col5:
        #     st.markdown("<h5 style='color: #008080; text-align:center'>Class 2: <b style='color:red'>D</b></h5>", unsafe_allow_html=True)
        #shap.summary_plot(shap_values, test_X, max_display=20)
        shap.summary_plot(shap_values, test_X, plot_type="bar")
        st.pyplot(plt.gcf())
        
        # TODO_ it's completed, but we will hide it untile study the interpretation of this plot
    # elif chart_type == "force":
    #     # feature = st.selectbox('Select feature for dependence plot:', X.columns)
    #     fig, ax = plt.subplots()
    #     shap.initjs()
    #     force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], test_X)
    #     shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    #     st.components.v1.html(shap_html, height=300)
    elif chart_type =="Summary":
        shap.summary_plot(shap_values, test_X)
        st.pyplot(plt.gcf())
    else:
        #all_features = [i for i in test_X.columns]
        #s#t.write(all_features)
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            feature = st.selectbox("Select Feature",["Posted_by", "Page", "length", "hashtag_count"], key="depend")
        shap.dependence_plot(str(feature), shap_values, test_X)
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

    model_select2 = st.sidebar.selectbox(" ", ["LightGBM", "XGBoost"], label_visibility="collapsed", key="second_select") 
    tabs = st.tabs([ "Bar Plot", "Summary Plot", "Dependence Plot"]) #, "Force Plot"
    with tabs[0]:
        if model_select2 == "XGBoost":
            train_test_scores_df2, selected_model, ex_time = ml_model("xgb")
            st.markdown(f"<h2 style='color: #008080; text-align:center'>Features importance using {model_select2} Model.</h2>", unsafe_allow_html=True)
            exe_time = shap_explainer_chart(selected_model, "Bar")
            
            #disply_shap_values(fi, "three")

        elif model_select2 == "LightGBM":
            train_test_scores_df2, selected_model, ex_time = ml_model("LighGBM")
            st.markdown(f"<h2 style='color: #008080; text-align:center'>Features importance using {model_select2} Model.</h2>", unsafe_allow_html=True)
            exe_time  = shap_explainer_chart(selected_model, "Bar")
            
            #disply_shap_values(fi, "four")
    
    with tabs[1]:
        if model_select2 == "XGBoost":
            train_test_scores_df2, selected_model, ex_time = ml_model("xgb")
            st.markdown(f"<h2 style='color: #008080; text-align:center'>Features importance using {model_select2} Model.</h2>", unsafe_allow_html=True)
            exe_time = shap_explainer_chart(selected_model, "Summary")
            st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")

            #disply_shap_values(fi, "one")
        elif model_select2 == "LightGBM":
            train_test_scores_df2, selected_model, ex_time = ml_model("LighGBM")
            st.markdown(f"<h2 style='color: #008080; text-align:center'>Features importance using {model_select2} Model.</h2>", unsafe_allow_html=True)
            exe_time = shap_explainer_chart(selected_model, "Summary")
            st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")

    with tabs[2]:
        if model_select2 == "XGBoost":
            train_test_scores_df2, selected_model, ex_time = ml_model("xgb")
            st.markdown(f"<h2 style='color: #008080; text-align:center'>Features importance using {model_select2} Model.</h2>", unsafe_allow_html=True)
            exe_time = shap_explainer_chart(selected_model, "Dependence")
            #st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")

            #disply_shap_values(fi, "one")
        elif model_select2 == "LightGBM":
            train_test_scores_df2, selected_model, ex_time = ml_model("LighGBM")
            st.markdown(f"<h2 style='color: #008080; text-align:center'>Features importance using {model_select2} Model.</h2>", unsafe_allow_html=True)
            exe_time = shap_explainer_chart(selected_model, "Dependence")
            #st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")
            #disply_shap_values(fi, "two")
        # elif model_select2 == "Randomforest":
        #     all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("rf")
        #     exe_time = shap_explainer_chart(selected_model, "Summary", 3)
        #     st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")
    

    # TODO_ Hide untill finish interpreting of plot
    # with tabs[2]:
    #     if model_select2 == "XGBoost":
    #         all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("xgb")
    #         exe_time = shap_explainer_chart(selected_model, "force", 6)
    #         # st.sidebar.success(f"SHAP execution time is: {exe_time:.2f} seconds")
    #     elif model_select2 == "LightGBM":
    #         all_scores_df2, train_test_scores_df2, selected_model, ex_time = ml_model("LighGBM")
    #         exe_time, fi  = shap_explainer_chart(selected_model, "force", 7)


