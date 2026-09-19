import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Credit Score Classification App",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_and_train_model():
    """Loads dataset, cleans data, performs feature engineering, and trains the model[cite: 1]."""
    if not os.path.exists("test.csv"):
        return None, None
    
    df = pd.read_csv("test.csv")
    
    def clean_numeric(series):
        return pd.to_numeric(series.astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce')

    # Data Cleaning steps from notebook[cite: 1]
    df['Age'] = clean_numeric(df['Age'])
    df['Annual_Income'] = clean_numeric(df['Annual_Income'])
    df['Num_of_Loan'] = clean_numeric(df['Num_of_Loan'])
    df['Num_of_Delayed_Payment'] = clean_numeric(df['Num_of_Delayed_Payment'])
    df['Changed_Credit_Limit'] = clean_numeric(df['Changed_Credit_Limit'])
    df['Outstanding_Debt'] = clean_numeric(df['Outstanding_Debt'])
    df['Amount_invested_monthly'] = clean_numeric(df['Amount_invested_monthly'])
    df['Monthly_Balance'] = clean_numeric(df['Monthly_Balance'])

    # Target Creation & Feature Engineering[cite: 1]
    df['Credit_Target'] = df['Credit_Mix'].apply(lambda x: 1 if str(x).strip() == 'Good' else 0)
    df['Debt_Income_Ratio'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1e-5)
    df['Total_Delinquency'] = df['Delay_from_due_date'] + df['Num_of_Delayed_Payment'].fillna(0)

    features = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Credit_Utilization_Ratio',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
        'Debt_Income_Ratio', 'Total_Delinquency'
    ]

    X = df[features].copy()
    y = df['Credit_Target']
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Model[cite: 1]
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "model.pkl")
    return model, features

# App UI Header
st.title("💳 Credit Score Classification Dashboard")
st.markdown("Predict whether a customer qualifies for a **Good** credit mix based on their financial and behavioral indicators[cite: 1].")

model, features = load_and_train_model()

if model is None:
    st.error("🚨 Dataset `test.csv` not found in the current working directory. Please upload it to proceed.")
else:
    st.sidebar.header("Customer Profile Inputs")

    # Sidebar inputs corresponding to model features
    age = st.sidebar.slider("Age", 18, 90, 30)
    annual_income = st.sidebar.number_input("Annual Income ($)", value=35000.0, step=1000.0)
    monthly_salary = st.sidebar.number_input("Monthly Inhand Salary ($)", value=3000.0, step=100.0)
    outstanding_debt = st.sidebar.number_input("Outstanding Debt ($)", value=500.0, step=50.0)
    num_bank_accounts = st.sidebar.slider("Number of Bank Accounts", 0, 20, 3)
    num_credit_card = st.sidebar.slider("Number of Credit Cards", 0, 15, 4)
    interest_rate = st.sidebar.slider("Interest Rate (%)", 0, 35, 5)
    num_of_loan = st.sidebar.slider("Number of Loans", 0, 10, 2)
    delay_from_due_date = st.sidebar.slider("Delay from Due Date (Days)", 0, 50, 5)
    num_delayed_payment = st.sidebar.slider("Number of Delayed Payments", 0.0, 30.0, 2.0)
    num_credit_inquiries = st.sidebar.slider("Number of Credit Inquiries", 0.0, 20.0, 3.0)
    credit_util_ratio = st.sidebar.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 30.0)
    total_emi = st.sidebar.number_input("Total EMI per Month ($)", value=50.0, step=10.0)
    amount_invested = st.sidebar.number_input("Amount Invested Monthly ($)", value=100.0, step=10.0)
    monthly_balance = st.sidebar.number_input("Monthly Balance ($)", value=250.0, step=25.0)

    # Compute derived features automatically
    debt_income_ratio = outstanding_debt / (annual_income + 1e-5)
    total_delinquency = delay_from_due_date + num_delayed_payment

    # Compile input data frame
    input_data = pd.DataFrame([[
        age, annual_income, monthly_salary, num_bank_accounts,
        num_credit_card, interest_rate, num_of_loan, delay_from_due_date,
        num_delayed_payment, num_credit_inquiries, credit_util_ratio,
        total_emi, amount_invested, monthly_balance, 
        debt_income_ratio, total_delinquency
    ]], columns=features)

    # Main Panel Prediction Section
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Review Input Data")
        st.dataframe(input_data.T, use_container_width=True)

    with col2:
        st.subheader("Prediction Action")
        if st.button("Predict Credit Status", type="primary", use_container_width=True):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.success(f"🌟 **Result: Good Credit Mix** \n\nConfidence: **{probability:.2%}**")
            else:
                st.warning(f"⚠️ **Result: Non-Good Credit Mix** \n\nConfidence: **{(1 - probability):.2%}**")