"""
Diabetes Prediction Gradio App
Deploy to Hugging Face Spaces
"""

import gradio as gr
import pickle
import numpy as np
import os

# ============================================
# Load Model and Scaler
# ============================================

try:
    # Try loading from current directory (Hugging Face)
    model = pickle.load(open("model/diabetes_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    feature_names = pickle.load(open("model/feature_names.pkl", "rb"))
    print("✓ Models loaded from current directory")
except:
    # Fallback to parent directory (local testing)
    try:
        model = pickle.load(open("../model/diabetes_model.pkl", "rb"))
        scaler = pickle.load(open("../model/scaler.pkl", "rb"))
        feature_names = pickle.load(open("../model/feature_names.pkl", "rb"))
        print("✓ Models loaded from parent directory")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        model = None
        scaler = None
        feature_names = None


# ============================================
# Prediction Function
# ============================================

def predict_diabetes(pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age):
    """
    Predict diabetes risk based on Pima Indian health metrics
    
    Args:
        pregnancies: Number of pregnancies (0-17)
        glucose: Blood glucose level (0-300 mg/dL)
        bp: Blood pressure (0-200 mm Hg)
        skin: Skin thickness (0-100 mm)
        insulin: Serum insulin level (0-900 mU/mL)
        bmi: Body mass index (10-60 kg/m²)
        pedigree: Diabetes pedigree function (0-2.5)
        age: Patient age (0-120 years)
    
    Returns:
        Formatted markdown string with prediction and confidence scores
    """
    
    if model is None or scaler is None:
        return "❌ **Error**: Model not loaded. Please ensure model files exist."
    
    try:
        # Create input array in correct feature order
        # Feature order MUST match training: [pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
        
        # Scale the input using the same scaler from training
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Extract confidence scores
        confidence_no_diabetes = probabilities[0] * 100
        confidence_diabetes = probabilities[1] * 100
        
        # Format output
        if prediction == 0:
            result = f"""
### ✅ Prediction Result: **LOW RISK (Negative)**

**Confidence Scores:**
- No Diabetes: **{confidence_no_diabetes:.2f}%**
- Diabetes: **{confidence_diabetes:.2f}%**

**Your Health Metrics:**
- Age: {age} years
- Pregnancies: {pregnancies}
- Glucose Level: {glucose} mg/dL
- Blood Pressure: {bp} mm Hg
- Skin Thickness: {skin} mm
- Insulin Level: {insulin} mU/mL
- BMI: {bmi} kg/m²
- Pedigree Function: {pedigree}

---

**⚠️ IMPORTANT DISCLAIMER:**
This prediction is **for educational purposes only** and should **NOT** be used for medical diagnosis or treatment decisions. 
Please consult with qualified healthcare professionals for proper medical evaluation.
"""
        else:
            result = f"""
### ⚠️ Prediction Result: **HIGH RISK (Positive)**

**Confidence Scores:**
- No Diabetes: **{confidence_no_diabetes:.2f}%**
- Diabetes: **{confidence_diabetes:.2f}%**

**Your Health Metrics:**
- Age: {age} years
- Pregnancies: {pregnancies}
- Glucose Level: {glucose} mg/dL
- Blood Pressure: {bp} mm Hg
- Skin Thickness: {skin} mm
- Insulin Level: {insulin} mU/mL
- BMI: {bmi} kg/m²
- Pedigree Function: {pedigree}

---

**⚠️ IMPORTANT DISCLAIMER:**
This prediction is **for educational purposes only** and should **NOT** be used for medical diagnosis or treatment decisions.
If results indicate elevated risk, **please consult with a healthcare professional immediately**.
"""
        
        return result
    
    except Exception as e:
        return f"❌ **Error during prediction**: {str(e)}"


# ============================================
# Gradio Interface
# ============================================

with gr.Blocks(title="Diabetes Prediction System", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🏥 Diabetes Prediction System
    
    **Using Logistic Regression with Pima Indian Diabetes Dataset**
    
    This application predicts the risk of diabetes based on health metrics.
    Enter your health information below and click "Predict" to get a risk assessment.
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Enter Your Health Information")
            
            pregnancies = gr.Slider(
                minimum=0,
                maximum=17,
                value=0,
                step=1,
                label="Number of Pregnancies",
                info="Total pregnancies (0-17)"
            )
            
            glucose = gr.Slider(
                minimum=0,
                maximum=300,
                value=100,
                step=1,
                label="Glucose Level (mg/dL)",
                info="Blood glucose level (0-300)"
            )
            
            bp = gr.Slider(
                minimum=0,
                maximum=200,
                value=70,
                step=1,
                label="Blood Pressure (mm Hg)",
                info="Diastolic blood pressure (0-200)"
            )
            
            skin = gr.Slider(
                minimum=0,
                maximum=100,
                value=20,
                step=1,
                label="Skin Thickness (mm)",
                info="Triceps skin fold (0-100)"
            )
            
            insulin = gr.Slider(
                minimum=0,
                maximum=900,
                value=80,
                step=1,
                label="Insulin Level (mU/mL)",
                info="Serum insulin (0-900)"
            )
            
            bmi = gr.Slider(
                minimum=10,
                maximum=60,
                value=25,
                step=0.1,
                label="BMI (kg/m²)",
                info="Body mass index (10-60)"
            )
            
            pedigree = gr.Slider(
                minimum=0,
                maximum=2.5,
                value=0.5,
                step=0.01,
                label="Diabetes Pedigree Function",
                info="Family history (0-2.5)"
            )
            
            age = gr.Slider(
                minimum=0,
                maximum=120,
                value=30,
                step=1,
                label="Age (years)",
                info="Your age (0-120)"
            )
            
            predict_button = gr.Button("🔍 Predict", size="lg", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 📋 Prediction Results")
            output = gr.Markdown("*Click 'Predict' to see results*")
    
    # Connect button to prediction function
    predict_button.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    
    ### ℹ️ About This Model
    
    - **Algorithm**: Logistic Regression (beginner-friendly ML classifier)
    - **Dataset**: Pima Indian Diabetes Dataset (768 samples)
    - **Features**: 8 health metrics combined to predict diabetes risk
    - **Model Type**: Binary Classification (Positive/Negative)
    - **Accuracy**: ~75% on test data
    
    ### 📚 Features Explained
    
    1. **Pregnancies**: Total number of times pregnant
    2. **Glucose**: Plasma glucose concentration
    3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
    4. **Skin Thickness**: Triceps skin fold thickness (mm)
    5. **Insulin**: 2-Hour serum insulin (mU/mL)
    6. **BMI**: Body mass index (weight in kg / height in m²)
    7. **Pedigree Function**: Diabetes pedigree function (genetic predisposition)
    8. **Age**: Age in years
    
    ### ⚠️ Important Disclaimer
    
    **This model is for educational and demonstration purposes only.**
    
    - NOT intended for medical diagnosis
    - NOT a substitute for professional medical advice
    - Should NOT be used to make health decisions
    - Always consult qualified healthcare professionals
    
    The predictions are based on a statistical model trained on historical data and may not apply to individual cases.
    """)


if __name__ == "__main__":
    demo.launch(share=True)
