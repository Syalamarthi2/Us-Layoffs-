import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the trained model and scalers/encoders
model = joblib.load("best_layoff_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model_columns = joblib.load("X_train_columns.pkl")  # Feature columns used in training

# Categorical columns
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
    df_input = pd.DataFrame([user_data])
    df_input = pd.get_dummies(df_input, columns=categorical_columns)

    for col in model_columns:
        if col not in df_input:
            df_input[col] = 0
    df_input = df_input[model_columns]

    df_scaled = scaler.transform(df_input)
    return df_scaled

# Load dataset for time-series analysis
@st.cache_data
def load_data():
    df = pd.read_csv("processed_layoff_data.csv")
    df['Separation Date'] = pd.to_datetime(df['Separation Date'], errors='coerce')
    df['Year'] = df['Separation Date'].dt.year
    df['Month'] = df['Separation Date'].dt.month
    df['Year-Month'] = df['Separation Date'].dt.to_period('M')
    return df

# UI Layout
st.title("🔍 Employee Layoff Prediction & Analysis")

# Tabs for switching between Prediction and Analysis
tab1, tab2 = st.tabs(["🧠 Predict Layoff Reason", "📈 Time-Series Analysis"])

# ================== PREDICTION TAB ==================
with tab1:
    st.header("📊 Enter Employee Details")

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

        processed_input = preprocess_input(user_data)
        prediction = model.predict(processed_input)
        predicted_reason = label_encoder.inverse_transform(prediction)[0]

        st.success(f"✅ **Predicted Layoff Reason:** {predicted_reason}")

# ================== TIME SERIES TAB ==================
with tab2:
    st.header("📅 Layoff Trends Over Time")

    df = load_data()

    # Monthly and Yearly Layoffs
    monthly_layoffs = df.groupby('Year-Month').size()
    yearly_layoffs = df.groupby('Year').size()

    st.subheader("📆 Monthly Layoffs Trend")
    st.line_chart(monthly_layoffs)

    st.subheader("📆 Yearly Layoffs Trend")
    st.line_chart(yearly_layoffs)

    # Additional: Employee Status Breakdown
    st.subheader("🧑‍💼 Employee Status Over Time")
    employee_status_monthly = df.groupby(['Year-Month', 'Employee Status']).size().unstack().fillna(0)
    st.line_chart(employee_status_monthly)

    # Optional: Interactive plot with matplotlib
    st.subheader("📉 Detailed Yearly Layoffs Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_layoffs.plot(kind='line', marker='o', color='darkorange', ax=ax)
    ax.set_title('Yearly Layoffs Trend')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Layoffs')
    ax.grid(True)
    st.pyplot(fig)
