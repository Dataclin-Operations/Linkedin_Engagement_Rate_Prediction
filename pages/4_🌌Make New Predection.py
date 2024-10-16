# #==========================================================================#
# #                          Import Libiraries
# #==========================================================================#
import sys

import streamlit as st
import pandas as pd
import os
import pandas as pd 
import numpy as np
import re
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from lightgbm import LGBMClassifier 

import warnings
# Suppress specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)




#==========================================================================#
#                          Import Data and functions From main file
#==========================================================================#

sys.path.append(os.path.abspath('../functions.py'))

from functions import ml_model, ml_model_predict, X






# columns = ["Drug","Age_In_Years","Sex","Ascites","Spiders","Edema","Bilirubin_log",
#            "Cholesterol_log","Albumin","Copper","Alk_Phos",
#            "SGOT_log","Tryglicerides_log","Platelets","Prothrombin","Stage","Mean_Metrics","AST_ALT_Ratio","Albumin_Bilirubin_Ratio","Age_Bilirubin","Albumin_Platelets","Child_Pugh_Proxy"]

#==========================================================================#
#                          Coolect Data Function
#==========================================================================#
# Initialize session state variables
# for key in ["Age_In_Years", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]:
#     if key not in st.session_state:
#         if key == "Age_In_Years":
#             st.session_state[key] = 0
#         else:
#             st.session_state[key] = 0.0

# def get_data_from_user():
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         Age_In_Years = st.number_input("Age", value=int(st.session_state.Age_In_Years), step=1, format="%d",placeholder="Patient Age",)
#         Bilirubin = st.number_input("Bilirubin", value=float(st.session_state.Bilirubin), step=1.0, format="%.2f", placeholder="Patient Bilirubin",)
#         Cholesterol = st.number_input("Cholesterol", value=float(st.session_state.Cholesterol), step=1.0, format="%.2f", placeholder="Patient Cholesterol")
#     with col2:
#         Albumin = st.number_input("Albumin", value=float(st.session_state.Albumin), step=1.0, format="%.2f", placeholder="Patient Albumin")

#         Copper = st.number_input("Copper", value=float(st.session_state.Copper), step=1.0, format="%.2f", placeholder="Patient Copper")
#         Alk_Phos = st.number_input("Alk_Phos", value=float(st.session_state.Alk_Phos), step=1.0, format="%.2f", placeholder="Patient Alk_Phos")

#     with col3:
#         SGOT = st.number_input("SGOT", value=float(st.session_state.SGOT), step=1.0, format="%.2f", placeholder="Patient SGOT")
#         Tryglicerides = st.number_input("Tryglicerides", value=float(st.session_state.Tryglicerides), step=1.0, format="%.2f", placeholder="Patient Tryglicerides")
#         Platelets = st.number_input("Platelets", value=float(st.session_state.Platelets), step=1.0, format="%.2f", placeholder="Patient Platelets")

#     with col1:
#         Prothrombin = st.number_input("Prothrombin", value=float(st.session_state.Prothrombin), step=1.0, format="%.2f", placeholder="Patient Prothrombin")
    
#     best_params = {"bagging_fraction": 0.8147515753137136, "feature_fraction": 0.7146909188772861, "min_child_samples": 94, "num_leaves": 92, "bagging_freq": 3, "learning_rate": 0.04702757643730806}
#     calculate = st.button("Predict Patient Outcome")
#     if calculate:
#         if Age_In_Years and Bilirubin  and Cholesterol and Albumin and Copper and Alk_Phos and SGOT and Tryglicerides and Platelets and Prothrombin:

#             data = {
#                 "Drug": [1],
#                 "Age_In_Years": [st.session_state.Age_In_Years],
#                 "Bilirubin": [st.session_state.Bilirubin],
#                 "Cholesterol": [st.session_state.Cholesterol],
#                 "Albumin": [st.session_state.Albumin],
#                 "Copper": [st.session_state.Copper],
#                 "Alk_Phos": [st.session_state.Alk_Phos],
#                 "SGOT": [st.session_state.SGOT],
#                 "Tryglicerides": [st.session_state.Tryglicerides],
#                 "Platelets": [st.session_state.Platelets],
#                 "Mean_Metrics": [0.50],
#                 "Prothrombin": [st.session_state.Prothrombin],
#             }
#             df = pd.DataFrame(data)
#             # 4.1.1 Flag for normal Bilirubin levels Bilirubin_Normal column
#             df["Bilirubin_Normal"] = df["Bilirubin"].apply(lambda x: 0 if x < 0.1 or x > 1.2 else 1).astype("int")

#             #4.1.2 Flag for normal Cholesterol levels Cholesterol_Normal column¶
#             df["Cholesterol_Normal"] = df["Cholesterol"].apply(lambda x: 0 if x > 200 else 1).astype("int")
            
#             #4.1.3 Flag for normal Albumin levels Albumin_Normal column¶
#             df["Albumin_Normal"] = df["Albumin"].apply(lambda x: 0 if x < 3.5 or x > 5 else 1).astype("int")

#             #4.1.4 Flag for normal Copper levels Copper_Normal column
#             df["Copper_Normal"] = df["Copper"].apply(lambda x: 0 if x < 70 or x > 150 else 1).astype("int")

#             #4.1.5 Flag for normal Alk_Phos levels Alk_Phos_Normal column
#             df["Alk_Phos_Normal"] = df["Alk_Phos"].apply(lambda x: 0 if x < 40 or x > 150 else 1).astype("int")

#             #4.1.6 Flag for normal SGOT levels SGOT_Normal column¶
#             df["SGOT_Normal"] = df["SGOT"].apply(lambda x: 0 if x < 8 or x > 40 else True).astype("int")

#             #4.1.7 Flag for normal Tryglicerides levels Tryglicerides_Normal column¶
#             df["Tryglicerides_Normal"] = df["Tryglicerides"].apply(lambda x: 0 if  x > 150 else 1).astype("int")

#             #4.1.8 Flag for normal Platelets levels Platelets_Normal column
#             df["Platelets_Normal"] = df["Platelets"].apply(lambda x: 0 if x < 150 or x > 450 else 1).astype("int")

#             #4.1.9 Flag for normal Prothrombin levels Prothrombin_Normal column¶
#             df["Prothrombin_Normal"] = df["Prothrombin"].apply(lambda x: 0 if x < 9.5 or x > 13.5 else 1).astype("int")

#             # Mean for metrics

#             #AST_ALT_Ratio
            
#             df['AST_ALT_Ratio'] = df['SGOT'] / df['Alk_Phos']

#             # Albumin to Bilirubin Ratio
#             df['Albumin_Bilirubin_Ratio'] = df['Albumin'] / df['Bilirubin']
#             # Interaction Terms
#             df['Age_Bilirubin'] = df['Age_In_Years'] * df['Bilirubin']
#             df['Albumin_Platelets'] = df['Albumin'] * df['Platelets']

#             # pugh score 

#             for col in ['Bilirubin', 'Cholesterol', 'SGOT', 'Tryglicerides']:
#                 df[col] = np.log1p(df[col])
#                 df.rename(columns={col: col+"_log"}, inplace=True)
        
#             X = df_ml[df.columns].copy()
#             y = df_ml.Status.copy()

#             # Split data 
#             random_state = 42
#             train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=random_state)

#             # Proportions of all classes
#             class_proportions = {0: 0.628083, 2: 0.337128, 1: 0.034788}

#             # Calculate class weights as inverse of proportions
#             class_weights = {cls: 1.0 / proportion for cls, proportion in class_proportions.items()}
        
#             model = LGBMClassifier(**best_params, random_state=random_state, class_weight=class_weights)
#             model.fit(train_X, train_y)
#             preds = model.predict(df)
#             target_reverse_mapping = {
#                 0:'C: (censored)',
#                 1:'CL: (alive due to liver transplant)',
#                 2:'D: (deceased)'
#             }
#             # st.write()
#             st.markdown(f"<h1 style='color: #008080; text-align:center'>The predicted Outcome for this patient is {target_reverse_mapping[preds[0]]}</h1>", unsafe_allow_html=True)

#         else:
#             st.error("Please fill all fields.")
#---------------------
# @st.cache_data()    
# def ml_model_run(dataf):
#             df = dataf.copy()
#             # 4.1.1 Flag for normal Bilirubin levels Bilirubin_Normal column
#             df["Bilirubin_Normal"] = df["Bilirubin"].apply(lambda x: 0 if x < 0.1 or x > 1.2 else 1).astype("int")

#             #4.1.2 Flag for normal Cholesterol levels Cholesterol_Normal column¶
#             df["Cholesterol_Normal"] = df["Cholesterol"].apply(lambda x: 0 if x > 200 else 1).astype("int")
            
#             #4.1.3 Flag for normal Albumin levels Albumin_Normal column¶
#             df["Albumin_Normal"] = df["Albumin"].apply(lambda x: 0 if x < 3.5 or x > 5 else 1).astype("int")

#             #4.1.4 Flag for normal Copper levels Copper_Normal column
#             df["Copper_Normal"] = df["Copper"].apply(lambda x: 0 if x < 70 or x > 150 else 1).astype("int")

#             #4.1.5 Flag for normal Alk_Phos levels Alk_Phos_Normal column
#             df["Alk_Phos_Normal"] = df["Alk_Phos"].apply(lambda x: 0 if x < 40 or x > 150 else 1).astype("int")

#             #4.1.6 Flag for normal SGOT levels SGOT_Normal column¶
#             df["SGOT_Normal"] = df["SGOT"].apply(lambda x: 0 if x < 8 or x > 40 else True).astype("int")

#             #4.1.7 Flag for normal Tryglicerides levels Tryglicerides_Normal column¶
#             df["Tryglicerides_Normal"] = df["Tryglicerides"].apply(lambda x: 0 if  x > 150 else 1).astype("int")

#             #4.1.8 Flag for normal Platelets levels Platelets_Normal column
#             df["Platelets_Normal"] = df["Platelets"].apply(lambda x: 0 if x < 150 or x > 450 else 1).astype("int")

#             #4.1.9 Flag for normal Prothrombin levels Prothrombin_Normal column¶
#             df["Prothrombin_Normal"] = df["Prothrombin"].apply(lambda x: 0 if x < 9.5 or x > 13.5 else 1).astype("int")

#             # Mean for metrics

#             #AST_ALT_Ratio
            
#             df['AST_ALT_Ratio'] = df['SGOT'] / df['Alk_Phos']

#             # Albumin to Bilirubin Ratio
#             df['Albumin_Bilirubin_Ratio'] = df['Albumin'] / df['Bilirubin']
#             # Interaction Terms
#             df['Age_Bilirubin'] = df['Age_In_Years'] * df['Bilirubin']
#             df['Albumin_Platelets'] = df['Albumin'] * df['Platelets']

#             # pugh score 

#             for col in ['Bilirubin', 'Cholesterol', 'SGOT', 'Tryglicerides']:
#                 df[col] = np.log1p(df[col])
#                 df.rename(columns={col: col+"_log"}, inplace=True)
        
#             X = df_ml[df.columns].copy()
#             y = df_ml.Status.copy()

#             best_params = {"bagging_fraction":0.8147515753137136,"feature_fraction":0.7146909188772861,"min_child_samples":94,"num_leaves":92,"bagging_freq":3,"learning_rate":0.04702757643730806}
#             # Split data 
#             random_state = 42
#             train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=random_state)

#             # Proportions of all classes
#             class_proportions = {0: 0.628083, 2: 0.337128, 1: 0.034788}

#             # Calculate class weights as inverse of proportions
#             class_weights = {cls: 1.0 / proportion for cls, proportion in class_proportions.items()}
        
#             model = LGBMClassifier(**best_params, random_state=random_state, class_weight=class_weights)
#             model.fit(train_X, train_y)
#             preds = model.predict(df)
#             preds_Proba = model.predict_proba(df)
#             preds_Proba = pd.DataFrame(preds_Proba)
#             target_reverse_mapping = {
#                 0:'C: (censored)',
#                 1:'CL: (alive due to liver transplant)',
#                 2:'D: (deceased)'
#             }
#             # st.write()
#             st.markdown(f"<h2 style='color: #008080; text-align:center; border: 2px solid red;border-radius: 30px;'>The predicted Outcome for this patient is {target_reverse_mapping[preds[0]]}</h2>", unsafe_allow_html=True)
#             st.markdown(f"<h6 style='color: #008080; text-align:center'></h6>", unsafe_allow_html=True)
#             st.markdown(f"<h3 style='color: #008080; text-align:center'>The predicted Outcome Probabilities are:</h3>", unsafe_allow_html=True)

#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.markdown(f"<h5 style='color: #008080; text-align:center'>C: (censored) {round(preds_Proba.loc[0,0]*100, 2) }%</h5>", unsafe_allow_html=True)
#             with col2:
#                 st.markdown(f"<h5 style='color: #008080; text-align:center'>CL: (alive due to liver transplant) {round(preds_Proba.loc[0,1]*100, 2) }%</h5>", unsafe_allow_html=True)
#             with col3:
#                 st.markdown(f"<h5 style='color: #008080; text-align:center'>D: (deceased) {round(preds_Proba.loc[0,2]*100, 2) }%</h5>", unsafe_allow_html=True)


Posted_by_dict2 = {
    "Eman Hamdy Enab": 1,
    "Mina Adly":2,
    "Mohamad Taha" : 3,
    "Ahmed Kamel AbdelRazek" :4
}
Page_dict2 = {
     "Dataclin_Group":1, 
     "DataClin_CRO":2, 
     "DataClin_MA":3, 
     "DataClin_SMO":4
 }
 
@st.fragment()
def get_data_from_user2(model):
    model = model
    if 'post' not in st.session_state:
        st.session_state.post = ''
    col1, col2, col3 = st.columns([1,0.5,1])
    with col1:
        Posted_by = st.selectbox("Select oprion", ["Eman Hamdy Enab", "Mina Adly","Mohamad Taha", "Ahmed Kamel AbdelRazek" ])
        eng_rate_yesterday = st.number_input("Engagment Yesterday", value=float(50.0), step=1.0, format="%.2f", placeholder="eng yest",)
    with col3:
        page_name = st.selectbox("Select oprion", ["Dataclin_Group", "DataClin_CRO","DataClin_MA", "DataClin_SMO" ])
        eng_rate_pre_week = st.number_input("Engagment Previous Week", value=float(50.0), step=1.0, format="%.2f", placeholder="eng pre week",)
        
    # col1, col2, col3 = st.columns([2,0.5,0.5])
    # with col1:
    
    post = st.text_input("Enter the post", value=st.session_state.post)
    st.session_state.post = post

        





    
    to_predict = pd.DataFrame(
        {   "Post_title":[""],
            "Posted_by":[0],
            "Page":[0],
            "length":[0],
            "hashtag_count":[0],
            "#ClinicalResearch":[0],
            "#ClinicalTrials":[0],
            "#hiring":[0],
            "#Role_and_respon":[0],
            "#MedicalResearch":[0],
            "#MedicalAdvancem":[0],
            "#Innovation":[0],
            "#Junior_General_":[0],
            "#MedicalWriting":[0],
            "lag_1":[0],
            "lag_7":[0],
            }
        )

    calualte = st.button("Predict Patient Outcome")
    if calualte:
        if (Posted_by and 
            page_name and
            eng_rate_yesterday and
            eng_rate_pre_week and
            post
                  ):
            
            to_predict["Post_title"] = post
            to_predict["Posted_by"] = Posted_by_dict2[Posted_by]
            to_predict["Page"] = Page_dict2[page_name]
            to_predict["length"] = len(page_name)

                        # Function to extract all hashtags using the regex
            def extract_hashtags(text):
                return re.findall(r'#\w+', text)

            hastags = to_predict["Post_title"].apply(extract_hashtags)
            to_predict["hashtag_count"] =  hastags.apply(lambda x : len(x))

            
            to_predict["#ClinicalResearch"] = to_predict["Post_title"].apply(lambda x: 1 if "#ClinicalResearch" in x else 0)
            to_predict["#ClinicalTrials"] = to_predict["Post_title"].apply(lambda x: 1 if "#ClinicalTrials" in x else 0)
            to_predict["#hiring"] = to_predict["Post_title"].apply(lambda x: 1 if "#hiring" in x else 0)
            to_predict["#Role_and_respon"] = to_predict["Post_title"].apply(lambda x: 1 if "#Role_and_respon" in x else 0)
            to_predict["#MedicalResearch"] = to_predict["Post_title"].apply(lambda x: 1 if "#MedicalResearch" in x else 0)
            to_predict["#MedicalAdvancem"] = to_predict["Post_title"].apply(lambda x: 1 if "#MedicalAdvancem" in x else 0)
            to_predict["#Innovation"] = to_predict["Post_title"].apply(lambda x: 1 if "#Innovation" in x else 0)
            to_predict["#Junior_General_"] = to_predict["Post_title"].apply(lambda x: 1 if "#Junior_General_" in x else 0)
            to_predict["#MedicalWriting"] = to_predict["Post_title"].apply(lambda x: 1 if "#MedicalWriting" in x else 0)
            to_predict["lag_1"] = eng_rate_yesterday
            to_predict["lag_7"] = eng_rate_pre_week



            to_predict = to_predict.drop('Post_title', axis=1)
            #st.write(to_predict)
            preds, execution_time = ml_model_predict(model, to_predict)
            st.markdown(f"<h1 style='color: #008080; text-align:center'>The predicted Engagment rate is {np.round(preds[0]*100, 2)} %</h1>", unsafe_allow_html=True)
            st.success(f"The Predection time is: {execution_time:.2f} seconds")
            new_X = pd.concat([X, to_predict],axis=0)
            shap.initjs()
            
            # explainer = shap.Explainer(selected_model, new_X)
            # #shap_values = explainer(X)
            # shap_values = explainer.shap_values(new_X)
            # instance_shap_values = shap_values[0]

            # plt.figure(figsize=(3, 1))
            # shap.plots.waterfall(instance_shap_values)
            # st.pyplot(plt.gcf())

            st.markdown(f"<h2 style='color: #008080; text-align:center'>SHAP Waterfall Plot for the new prediction</h2>", unsafe_allow_html=True)
            explainer = shap.Explainer(selected_model, new_X)

            # Compute SHAP values (this will return an Explanation object)
            shap_values = explainer(new_X)

            # Select the SHAP values for the last instance (which is an Explanation object)
            instance_shap_values = shap_values[-1]

            # Ensure that we are passing the correct object to the waterfall plot
            if isinstance(instance_shap_values, shap.Explanation):
                plt.figure(figsize=(3, 1))
                shap.plots.waterfall(instance_shap_values)
                
                # Show the plot in Streamlit
                st.pyplot(plt.gcf())
        else:
            st.error("Please fill all fields.")
   

#==========================================================================#
#                          Main App
#==========================================================================#


if __name__ == "__main__":
    logo_path = "./TheLogo2.png"  # Replace with the path to your logo image
    st.sidebar.image(logo_path, use_column_width=True)
    st.sidebar.markdown("<h2 style='color: #008080; text-align:center'>Select a Model</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        model_select = st.sidebar.selectbox(" ", ["LightGBM", "XGBoost"], label_visibility="collapsed")    
    if model_select == "XGBoost":
        train_test_scores_df2, selected_model, ex_time = ml_model("xgb")
    elif model_select == "LightGBM":
        train_test_scores_df2, selected_model, ex_time = ml_model("LighGBM")
   
    st.sidebar.success(f"The Model execution time is: {ex_time:.2f} seconds")

    get_data_from_user2(selected_model)
    # get_data_from_user()

   