import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="HR Attrition Prediction", layout="wide")
st.title("👩‍💼 HR Attrition Prediction App")
st.write("Upload employee data to predict if the employee is at risk of leaving.")

# ✅ Load and train the model once
@st.cache_resource
def load_model():
    df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    # Drop irrelevant columns
    df.drop(['Over18', 'EmployeeNumber', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True)

    # Encode target
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Split
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns.tolist()

model, expected_columns = load_model()

# 📤 File uploader
uploaded_file = st.file_uploader("Upload a CSV file with employee data", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(data.head())

    try:
        # Ensure the input matches expected features
        input_data = pd.get_dummies(data)
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)

        # Predict
        predictions = model.predict(input_data)
        data["Attrition Prediction"] = predictions
        st.subheader("🎯 Prediction Results")
        st.dataframe(data)

    except Exception as e:
        st.error("❌ Error: Check if your uploaded file matches the expected format.")
        st.exception(e)
