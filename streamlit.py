import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function to load the trained model
def load_model():
    return pickle.load(open('trained_model.sav', 'rb'))

# Function to preprocess input data
def preprocess_input_data(input_data):
    le = LabelEncoder()
    for obj in input_data.select_dtypes(include=['object']).columns:
        input_data[obj] = le.fit_transform(input_data[obj])
    return input_data

# Function to make prediction
def make_prediction(model, input_data):
    y_pred = model.predict(input_data.values)
    return y_pred

# Function to display result
def display_result(y_pred):
    if st.button('SUBMIT'):
        if y_pred == 0:
            st.success('It is recommended to consider seeking treatment.')
        else:
            st.warning('Great news! Treatment is not needed.')

# Main function
def main():

    st.title("MENTAL TREATMENT")

    # Introduction
    st.write("Welcome to our Mental Health Assessment Tool!")
    st.write("This tool is built using a machine learning model.")
    st.write("It aims to provide insights into whether seeking mental health treatment may be beneficial in a workplace context.")    

    # Load the trained model
    loaded_model = load_model()

    # User input section
    age = st.slider("Age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender",['Female', 'Male', 'Other'])
    family_history = st.selectbox("Do you have a family history of mental illness?",
                                ['No' ,'Yes'])
    work_interfere = st.selectbox("If you have a mental health condition, do you feel that it interferes with your work?",
                                ['Often' ,'Rarely' ,'Never' ,'Sometimes' ,"Don't know"])
    no_employees = st.selectbox("How many employees does your company or organization have?",
                                ['6-25' ,'More than 1000' ,'26-100' ,'100-500' ,'1-5' ,'500-1000'])

    tech_company = st.selectbox("Is your employer primarily a tech company/organization?",
                                 ['Yes','No'])
    benefits = st.selectbox("Does your employer provide mental health benefits?",
                                  ['Yes', "Don't know" ,'No'])
    care_options = st.selectbox("Do you know the options for mental health care your employer provides?",
                                  ['Not sure', 'No', 'Yes'])
    wellness_program = st.selectbox("Has your employer ever discussed mental health as part of an employee wellness program?",
                                  ['No', "Don't know" ,'Yes'])
    seek_help = st.selectbox('Does your employer provide resources to learn more about mental health issues and how to seek help?',
                                  ['Yes' ,"Don't know", 'No'])
    anonymity = st.selectbox('Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?',
                               ['Yes' ,"Don't know", 'No'])
    leave = st.selectbox('How easy is it for you to take medical leave for a mental health condition?',
                        ['Somewhat easy', "Don't know" ,'Somewhat difficult', 'Very difficult' ,'Very easy'])                        
    coworkers = st.selectbox('Would you be willing to discuss a mental health issue with your coworkers?',
                        ['Some of them' 'No' 'Yes'])
    supervisor = st.selectbox('Would you be willing to discuss a mental health issue with your direct supervisor(s)?',
                        ['Yes','No','Some of them'])
    mental_health_interview = st.selectbox('Would you bring up a mental health issue with a potential employer in an interview?',
                                  ['No' ,'Yes' ,'Maybe'])
    phys_health_interview = st.selectbox('Would you bring up a physical health issue with a potential employer in an interview?',
                        ['Maybe', 'No' ,'Yes'])
    mental_vs_physical = st.selectbox('Do you feel that your employer takes mental health as seriously as physical health?',
                        ['Yes' ,"Don't know" ,'No'])
    obs_consequence = st.selectbox('Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?',
                                  ['No' ,'Yes'])

    input_data = pd.DataFrame({
            "Age" : [age],
             "Gender" : [gender],
             "Do you have a family history of mental illness?" : [family_history],
             "If you have a mental health condition, do you feel that it interferes with your work?" : [work_interfere],
             "How many employees does your company or organization have?" : [no_employees],
             "Is your employer primarily a tech company/organization?" : [tech_company],
             "Does your employer provide mental health benefits?" : [benefits],
             "Do you know the options for mental health care your employer provides?" : [care_options],
             "Has your employer ever discussed mental health as part of an employee wellness program?" : [wellness_program],
             'Does your employer provide resources to learn more about mental health issues and how to seek help?' : [seek_help],
             'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?' : [anonymity],
             'How easy is it for you to take medical leave for a mental health condition?' : [leave],
             'Would you be willing to discuss a mental health issue with your coworkers?' : [coworkers],
             'Would you be willing to discuss a mental health issue with your direct supervisor(s)?' : [supervisor],
             'Would you bring up a mental health issue with a potential employer in an interview?' : [mental_health_interview],
             'Would you bring up a physical health issue with a potential employer in an interview?' : [phys_health_interview],
             'Do you feel that your employer takes mental health as seriously as physical health?' : [mental_vs_physical],
             'Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?' : [obs_consequence]
                          })
    # Preprocess input data
    input_data = preprocess_input_data(input_data)

    # Make prediction
    y_pred = make_prediction(loaded_model, input_data)

    # Display result
    display_result(y_pred)

if __name__ == "__main__":
    main()
