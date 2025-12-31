import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Liver Health AI", page_icon="🩺", layout="centered")

# --- DEBUGGING / LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Check if files exist to avoid silent blank screen
        if not os.path.exists('liver_disease_pipeline.pkl'):
            return None, None, "Error: 'liver_disease_pipeline.pkl' not found on GitHub."
        if not os.path.exists('feature_names.pkl'):
            return None, None, "Error: 'feature_names.pkl' not found on GitHub."
            
        model = joblib.load('liver_disease_pipeline.pkl')
        features = joblib.load('feature_names.pkl')
        return model, features, "Success"
    except Exception as e:
        return None, None, str(e)

# Initialize
pipeline, feature_names, status_message = load_assets()

# --- IF LOADING FAILED, SHOW ERROR INSTEAD OF BLANK SCREEN ---
if pipeline is None:
    st.error("### ⚠️ App Initialization Failed")
    st.write(f"**Details:** {status_message}")
    st.info("Ensure you uploaded the .pkl files to the main folder of your GitHub repository.")
    st.stop()

# --- UI DESIGN ---
st.title("🩺 Clinical Liver Disease Predictor")
st.markdown("Enter laboratory metrics to evaluate liver health status.")
st.divider()

# Organize inputs into two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 40)
    sex = st.selectbox("Sex", ["m", "f"])
    alb = st.number_input("Albumin (ALB)", value=35.0)
    alp = st.number_input("Alkaline Phosphatase (ALP)", value=70.0)
    alt = st.number_input("Alanine Aminotransferase (ALT)", value=20.0)
    ast = st.number_input("Aspartate Aminotransferase (AST)", value=30.0)

with col2:
    bil = st.number_input("Bilirubin (BIL)", value=10.0)
    che = st.number_input("Cholinesterase (CHE)", value=7.0)
    chol = st.number_input("Cholesterol (CHOL)", value=4.5)
    crea = st.number_input("Creatinina (CREA)", value=80.0)
    ggt = st.number_input("GGT", value=35.0)
    prot = st.number_input("Total Protein (PROT)", value=70.0)

# --- PREDICTION ---
if st.button("Generate Diagnostic Report", type="primary", use_container_width=True):
    # Prepare Data
    input_data = {
        'age': age, 'sex': sex, 'albumin': alb, 'alkaline_phosphatase': alp,
        'alanine_aminotransferase': alt, 'aspartate_aminotransferase': ast,
        'bilirubin': bil, 'cholinesterase': che, 'cholesterol': chol,
        'creatinina': crea, 'gamma_glutamyl_transferase': ggt, 'protein': prot
    }
    
    df = pd.DataFrame([input_data])
    
    # Preprocessing (Matches notebook mapping)
    df['sex'] = df['sex'].map({'m': 1, 'f': 0}).fillna(1)
    
    # Ensure Column Order
    df = df[feature_names]
    
    # Predict
    pred_idx = pipeline.predict(df)[0]
    probs = pipeline.predict_proba(df)
    confidence = np.max(probs) * 100
    
    # Labels (Matches notebook target_names)
    diag_map = {0: "Healthy (Blood Donor)", 1: "Cirrhosis", 2: "Hepatitis", 3: "Fibrosis", 4: "Suspect Donor"}
    diagnosis = diag_map.get(pred_idx, "Unknown Stage")
    
    # Results Display
    st.divider()
    if pred_idx == 0:
        st.success(f"### Result: {diagnosis}")
    else:
        st.error(f"### Result: {diagnosis}")
        
    st.write(f"**Model Confidence:** {round(confidence, 2)}%")
    
    if pred_idx > 0:
        st.warning("⚠️ Clinical markers suggest liver abnormality. Professional consultation required.")
