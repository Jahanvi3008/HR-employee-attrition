import streamlit as st
import pandas as pd
import joblib

st.title("👩‍💼 HR Attrition Prediction App")
st.write("Upload employee data to predict if the employee is at risk of leaving.")

model = joblib.load("attrition_model.pkl")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    predictions = model.predict(data)
    data['Attrition Prediction'] = predictions
    st.write("Prediction Results:")
    st.dataframe(data)
