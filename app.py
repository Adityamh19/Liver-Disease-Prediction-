import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Liver Disease AI Diagnostic", page_icon="🩺", layout="centered")

# --- ASSET LOADING WITH ERROR HANDLING ---
@st.cache_resource
def load_liver_assets():
    try:
        # Check if the files actually exist in the GitHub repo
        if not os.path.exists('liver_disease_pipeline.pkl'):
            return None, None, "File 'liver_disease_pipeline.pkl' not found. Please upload it to GitHub."
        if not os.path.exists('feature_names.pkl'):
            return None, None, "File 'feature_names.pkl' not found. Please upload it to GitHub."
            
        model = joblib.load('liver_disease_pipeline.pkl')
        features = joblib.load('feature_names.pkl')
        return model, features, "Success"
    except Exception as e:
        return None, None, f"Loading Error: {str(e)}"

# Load the model and feature list
pipeline, feature_names, load_status = load_liver_assets()

# --- BLANK SCREEN PREVENTION ---
if pipeline is None:
    st.error("### ⚠️ Application Initialization Failed")
    st.info(load_status)
    st.stop()

# --- USER INTERFACE ---
st.title("🩺 Clinical Liver Disease Predictor")
st.markdown("Enter laboratory blood markers to evaluate the patient's liver health status.")
st.divider()

# Sidebar for inputs (more organized for medical data)
st.sidebar.header("Patient Data Input")

def get_user_inputs():
    age = st.sidebar.slider("Age", 1, 100, 45)
    sex = st.sidebar.selectbox("Sex", ["m", "f"])
    
    col1, col2 = st.columns(2)
    with col1:
        alb = st.number_input("Albumin (ALB)", value=35.0)
        alp = st.number_input("Alkaline Phosphatase (ALP)", value=70.0)
        alt = st.number_input("Alanine Aminotransferase (ALT)", value=25.0)
        ast = st.number_input("Aspartate Aminotransferase (AST)", value=30.0)
        bil = st.number_input("Bilirubin (BIL)", value=10.0)
    
    with col2:
        che = st.number_input("Cholinesterase (CHE)", value=8.0)
        chol = st.number_input("Cholesterol (CHOL)", value=5.0)
        crea = st.number_input("Creatinina (CREA)", value=80.0)
        ggt = st.number_input("GGT", value=40.0)
        prot = st.number_input("Total Protein (PROT)", value=70.0)

    return {
        'age': age, 'sex': sex, 'albumin': alb, 'alkaline_phosphatase': alp,
        'alanine_aminotransferase': alt, 'aspartate_aminotransferase': ast,
        'bilirubin': bil, 'cholinesterase': che, 'cholesterol': chol,
        'creatinina': crea, 'gamma_glutamyl_transferase': ggt, 'protein': prot
    }

user_data = get_user_inputs()

# --- PREDICTION LOGIC ---
if st.button("Analyze & Predict", type="primary", use_container_width=True):
    # Create DataFrame
    df = pd.DataFrame([user_data])
    
    # 1. Map Sex (Matches your notebook mapping)
    df['sex'] = df['sex'].map({'m': 1, 'f': 0}).fillna(1)
    
    # 2. Reorder columns to match the training set exactly
    df = df[feature_names]
    
    # 3. Predict using the full Pipeline
    pred_idx = pipeline.predict(df)[0]
    prob = np.max(pipeline.predict_proba(df)) * 100
    
    # 4. Map back to Clinical Labels
    labels = {0: "Healthy (Blood Donor)", 1: "Cirrhosis", 2: "Hepatitis", 3: "Fibrosis", 4: "Suspect Donor"}
    result_name = labels.get(pred_idx, "Unknown")

    # --- DISPLAY RESULTS ---
    st.divider()
    if pred_idx == 0:
        st.success(f"### Predicted Status: {result_name}")
    else:
        st.error(f"### Predicted Status: {result_name}")
        
    st.metric("Model Confidence Level", f"{round(prob, 2)}%")
    
    if pred_idx != 0:
        st.warning("⚠️ Recommendation: This patient markers indicate a high risk of liver pathology. Immediate medical follow-up is suggested.")
