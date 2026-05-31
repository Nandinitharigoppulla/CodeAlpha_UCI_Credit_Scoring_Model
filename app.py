import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="Credit Scoring", layout="wide", page_icon="💳")
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button { background-color: #4CAF50; color: white; font-size: 18px; border-radius: 8px; }
    .pred-box { padding: 20px; border-radius: 10px; background-color: #e9ecef; text-align: center; }
    .risk-high { color: #d9534f; font-weight: bold; }
    .risk-low { color: #5cb85c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Scoring Model")
st.markdown("Predict the probability of default based on financial history.")

# ------------------------------
# Load model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load('models/credit_pipeline.pkl')

model = load_model()

# ------------------------------
# Input form
# ------------------------------
with st.form("credit_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        limit_bal = st.number_input("LIMIT_BAL (credit limit)", min_value=0, value=50000, step=10000)
        sex = st.selectbox("SEX", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        education = st.selectbox("EDUCATION", [1,2,3,4,5,6], 
                                 format_func=lambda x: {1:"Graduate",2:"University",3:"High School",4:"Others",5:"Unknown",6:"Unknown"}[x])
        marriage = st.selectbox("MARRIAGE", [1,2,3], format_func=lambda x: {1:"Married",2:"Single",3:"Others"}[x])
        age = st.number_input("AGE", min_value=18, max_value=100, value=30)

    with col2:
        pay_0 = st.selectbox("PAY_0 (Sept repayment)", [-2,-1,0,1,2,3,4,5,6,7,8])
        pay_2 = st.selectbox("PAY_2 (Aug)", [-2,-1,0,1,2,3,4,5,6,7,8])
        pay_3 = st.selectbox("PAY_3 (Jul)", [-2,-1,0,1,2,3,4,5,6,7,8])
        pay_4 = st.selectbox("PAY_4 (Jun)", [-2,-1,0,1,2,3,4,5,6,7,8])
        pay_5 = st.selectbox("PAY_5 (May)", [-2,-1,0,1,2,3,4,5,6,7,8])
        pay_6 = st.selectbox("PAY_6 (Apr)", [-2,-1,0,1,2,3,4,5,6,7,8])

    with col3:
        bill_amt1 = st.number_input("BILL_AMT1 (Sept bill)", value=0)
        bill_amt2 = st.number_input("BILL_AMT2 (Aug bill)", value=0)
        bill_amt3 = st.number_input("BILL_AMT3 (Jul bill)", value=0)
        bill_amt4 = st.number_input("BILL_AMT4 (Jun bill)", value=0)
        bill_amt5 = st.number_input("BILL_AMT5 (May bill)", value=0)
        bill_amt6 = st.number_input("BILL_AMT6 (Apr bill)", value=0)
        
    st.subheader("Payment amounts")
    col4, col5 = st.columns(2)
    with col4:
        pay_amt1 = st.number_input("PAY_AMT1 (Sept payment)", value=0)
        pay_amt2 = st.number_input("PAY_AMT2 (Aug payment)", value=0)
        pay_amt3 = st.number_input("PAY_AMT3 (Jul payment)", value=0)
    with col5:
        pay_amt4 = st.number_input("PAY_AMT4 (Jun payment)", value=0)
        pay_amt5 = st.number_input("PAY_AMT5 (May payment)", value=0)
        pay_amt6 = st.number_input("PAY_AMT6 (Apr payment)", value=0)

    submitted = st.form_submit_button("🔍 Predict Credit Risk")

# ------------------------------
# Prediction logic
# ------------------------------
if submitted:
    # Build input dataframe
    input_dict = {
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3,
        'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3,
        'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6
    }
    
    # Feature engineering (same as training)
    bill_vals = [bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6]
    pay_vals = [pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]
    pay_status = [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
    
    input_dict['avg_bill'] = np.mean(bill_vals)
    input_dict['avg_pay'] = np.mean(pay_vals)
    input_dict['pay_ratio'] = input_dict['avg_pay'] / (input_dict['avg_bill'] + 1e-6)
    input_dict['delinquent_count'] = sum(1 for x in pay_status if x > 0)
    input_dict['total_bill'] = sum(bill_vals)
    input_dict['total_pay'] = sum(pay_vals)
    
    input_df = pd.DataFrame([input_dict])
    
    # Prediction
    proba = model.predict_proba(input_df)[0, 1]
    pred_class = 1 if proba >= 0.5 else 0
    
    # Display results
    st.markdown("---")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.markdown(f"<div class='pred-box'><h3>Default Probability</h3><h1>{proba:.2%}</h1></div>", unsafe_allow_html=True)
    with col_res2:
        if pred_class == 1:
            st.markdown("<div class='pred-box risk-high'><h3>⚠️ High Risk</h3><h4>Likely to default</h4></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pred-box risk-low'><h3>✅ Low Risk</h3><h4>Creditworthy</h4></div>", unsafe_allow_html=True)
    
    st.progress(float(proba))# Probability bar
    
    
    # Feature importance (if needed)
    if st.checkbox("Show feature importance (from model)"):
        try:
            importance = model.named_steps['model'].feature_importances_
            preprocessor = model.named_steps['preprocessor']
            # Get feature names after preprocessing (simplified)
            st.write("Feature importance chart (top 10)")
            # We'll extract from model's internal structure – simplified here
            st.info("Feature importance available after training; for brevity skip detailed plot.")
        except:
            st.info("Feature importance not available in this pipeline view.")