import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("best_layoff_model.pkl")
scaler = joblib.load("scaler.pkl")  
label_encoder = joblib.load("label_encoder.pkl")  

# List of categorical columns from training dataset
categorical_columns = [
    "Department Name",
    "Sub-Unit of Department",
    "Employee Status",
    "Pay Plan",
    "Classification Title",
    "Voluntary/Involuntary",
    "EEO_EEO Category Name"
]

# Function to preprocess user input
def preprocess_input(user_data):
    # Convert input to DataFrame
    df_input = pd.DataFrame([user_data])

    # One-hot encode categorical features
    df_input = pd.get_dummies(df_input, columns=categorical_columns)

    # Load original column names from training (to align features)
    model_columns = joblib.load("X_train_columns.pkl")  # Saved original feature names
    for col in model_columns:
        if col not in df_input:
            df_input[col] = 0  # Add missing columns

    df_input = df_input[model_columns]  # Ensure column order is correct

    # Scale numerical features
    df_scaled = scaler.transform(df_input)

    return df_scaled

# Streamlit UI
st.title("🔍 Employee Layoff Prediction")

# User Inputs
record_number = st.number_input("Record Number", min_value=1000, max_value=99999, step=1)
fiscal_year = st.number_input("Fiscal Year", min_value=2000, max_value=2100, step=1)
pay_grade = st.number_input("Pay Grade", min_value=1.0, max_value=20.0, step=0.1)
separation_year = st.number_input("Separation Year", min_value=2000, max_value=2100, step=1)
current_fiscal_year = st.radio("Is it the Current Fiscal Year?", [1, 0])
recent_layoff = st.radio("Was there a Recent Layoff?", [1, 0])

# Categorical Inputs
department_name = st.selectbox("Department Name", ["Human Resources", "Finance", "IT", "Sales", "Operations"])
sub_unit = st.selectbox("Sub-Unit of Department", ["HR Administration", "Accounting", "Software", "Marketing"])
employee_status = st.selectbox("Employee Status", ["Permanent", "Temporary"])
pay_plan = st.selectbox("Pay Plan", ["General", "Special"])
classification_title = st.selectbox("Classification Title", ["HR Specialist", "Financial Analyst", "Software Engineer"])
voluntary_involuntary = st.selectbox("Voluntary/Involuntary", ["Voluntary", "Involuntary"])
eeo_category = st.selectbox("EEO Category", ["Administrative Support", "Official/Administrator", "Professional"])

# Make prediction when user clicks button
if st.button("🔮 Predict Layoff Reason"):
    user_data = {
        "Record Number": record_number,
        "Fiscal Year": fiscal_year,
        "Pay Grade": pay_grade,
        "Separation Year": separation_year,
        "Current Fiscal Year": current_fiscal_year,
        "Recent Layoff": recent_layoff,
        "Department Name": department_name,
        "Sub-Unit of Department": sub_unit,
        "Employee Status": employee_status,
        "Pay Plan": pay_plan,
        "Classification Title": classification_title,
        "Voluntary/Involuntary": voluntary_involuntary,
        "EEO_EEO Category Name": eeo_category,
    }

    # Preprocess user input
    processed_input = preprocess_input(user_data)

    # Make prediction
    prediction = model.predict(processed_input)
    predicted_reason = label_encoder.inverse_transform(prediction)[0]  # Convert back to category

    # Display result
    st.success(f"Predicted Layoff Reason: **{predicted_reason}**")
