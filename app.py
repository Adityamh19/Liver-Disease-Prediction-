import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="Liver Health AI", layout="centered")

@st.cache_resource
def load_liver_model():
    try:
        # These must match your GitHub filenames exactly
        model = joblib.load('liver_disease_pipeline.pkl')
        features = joblib.load('feature_names.pkl')
        return model, features
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

pipeline, feature_names = load_liver_model()

# --- 2. UI DESIGN ---
st.title("🩺 Clinical Liver Disease Predictor")
st.write("Professional Diagnostic Support System")
st.divider()

if pipeline is not None:
    # Creating a neat grid for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        sex = st.selectbox("Sex", ["m", "f"])
        alb = st.number_input("Albumin (ALB)", 10.0, 70.0, 35.0)
        alp = st.number_input("Alkaline Phosphatase (ALP)", 10.0, 500.0, 70.0)
        alt = st.number_input("Alanine Aminotransferase (ALT)", 1.0, 300.0, 25.0)
        ast = st.number_input("Aspartate Aminotransferase (AST)", 1.0, 300.0, 30.0)

    with col2:
        bil = st.number_input("Bilirubin (BIL)", 0.1, 300.0, 10.0)
        che = st.number_input("Cholinesterase (CHE)", 1.0, 20.0, 8.0)
        chol = st.number_input("Cholesterol (CHOL)", 1.0, 15.0, 5.0)
        crea = st.number_input("Creatinina (CREA)", 10.0, 1100.0, 80.0)
        ggt = st.number_input("GGT", 1.0, 700.0, 40.0)
        prot = st.number_input("Total Protein (PROT)", 30.0, 100.0, 70.0)

    # --- 3. PREDICTION ---
    if st.button("Generate Diagnostic Report", type="primary", use_container_width=True):
        input_data = {
            'age': age, 'sex': sex, 'albumin': alb, 'alkaline_phosphatase': alp,
            'alanine_aminotransferase': alt, 'aspartate_aminotransferase': ast,
            'bilirubin': bil, 'cholinesterase': che, 'cholesterol': chol,
            'creatinina': crea, 'gamma_glutamyl_transferase': ggt, 'protein': prot
        }
        
        df = pd.DataFrame([input_data])
        df['sex'] = df['sex'].map({'m': 1, 'f': 0}).fillna(1)
        
        # Reorder to match training
        df = df[feature_names]
        
        # Predict
        prediction = pipeline.predict(df)[0]
        prob = np.max(pipeline.predict_proba(df)) * 100
        
        labels = {0: "Healthy (Blood Donor)", 1: "Cirrhosis", 2: "Hepatitis", 3: "Fibrosis", 4: "Suspect Donor"}
        
        st.divider()
        if prediction == 0:
            st.success(f"### Result: {labels[prediction]}")
        else:
            st.error(f"### Result: {labels[prediction]}")
            
        st.metric("Confidence Level", f"{prob:.2f}%")
