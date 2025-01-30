import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("churn_data.csv")

st.markdown("<h1 style='text-align: center; color: #FF5733;'>Customer Churn Prediction System</h1>", unsafe_allow_html=True)

gender = st.selectbox("Select Gender", ['Female', 'Male'])
SeniorCitizen = st.selectbox("Are you a senior citizen?", ['Yes', 'No'])
Partner = st.selectbox("Do you have a partner?", ['Yes', 'No'])
Dependents = st.selectbox("Are you dependent on someone?", ['Yes', 'No'])
tenure = st.number_input("Enter your tenure (months)", min_value=0, max_value=100, step=1)
PhoneService = st.selectbox("Do you have phone service?", ['Yes', 'No'])
MultipleLines = st.selectbox("Do you have multiple lines?", ['Yes', 'No', 'No phone service'])
Contract = st.selectbox("Your contract type?", ['One year', 'Two year', 'Month-to-month'])
TotalCharges = st.number_input("Enter your total charges", min_value=0.0, format="%.2f")

def predict_churn(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges):
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'Contract': [Contract],
        'TotalCharges': [TotalCharges]
    })

    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract']
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        input_data[column] = label_encoder.fit_transform(input_data[column])

    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)
    return prediction[0]

if st.button("Predict Churn"):
    result = predict_churn(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges)

    if result == 1:
        st.markdown("<h2 style='color: red; text-align: center;'>⚠️ This customer is likely to CHURN! ⚠️</h2>", unsafe_allow_html=True)
        st.error("High risk of churn. Consider retention strategies!")
    else:
        st.markdown("<h2 style='color: green; text-align: center;'>✅ This customer is NOT likely to churn!</h2>", unsafe_allow_html=True)
        st.success("Low risk of churn. Keep engaging with the customer!")

