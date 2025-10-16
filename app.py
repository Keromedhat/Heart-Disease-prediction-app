import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# ---------------- CSS for Dark Theme + Style ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #FAFAFA;
}

h1, h2, h3, h4, h5, h6 {
    color: #FF6B6B;
}

.stButton>button {
    background-color: #FF6B6B;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
}

div.stNumberInput, div.stSelectbox {
    background-color: #1E1E2E;
    border-radius: 12px;
    padding: 5px;
    color: white;
}

.stMarkdown {
    color: #FAFAFA;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("## 🫀 Heart Disease Prediction App")
st.markdown("Enter the patient's details below to predict the likelihood of heart disease using the trained model.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load('best_xgboost_model.pkl')
    return model

model = load_model()

# ---------------- Input Fields ----------------
st.markdown("### 🧍‍♂️ Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age (العمر) — 20–60 yrs", 1, 120, 50)
    sex = st.selectbox("🚻 Sex (0 = Female, 1 = Male) / الجنس", [0, 1])
    cp = st.number_input("💔 Chest Pain Type (0–3)", 0, 3, 1)
    trestbps = st.number_input("🩺 Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol = st.number_input("💉 Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.number_input("📈 Resting ECG Results (0–2)", 0, 2, 1)
    thalch = st.number_input("❤️ Max Heart Rate (bpm)", 60, 220, 150)
    exang = st.selectbox("🏃 Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("📉 Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    slope = st.number_input("📊 Slope of ST Segment (0–2)", 0, 2, 1)
    ca = st.number_input("🫀 Number of Major Vessels (0–3)", 0, 3, 0)
    thal = st.number_input("🧬 Thalassemia (0=Normal,1=Fixed,2=Reversible)", 0, 2, 1)

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("### 📊 Prediction Result")

    if prob < 40:
        st.success(f"🩵 Low Risk ({prob:.2f}%) — Likely healthy.")
        color = "green"
    elif 40 <= prob <= 70:
        st.warning(f"🟡 Medium Risk ({prob:.2f}%) — May need checkup.")
        color = "yellow"
    else:
        st.error(f"🔴 High Risk ({prob:.2f}%) — Consult a doctor.")
        color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={'reference': 50, 'increasing': {'color': "red"}},
        title={'text': "Heart Disease Risk (%)", 'font': {'color': "white", 'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "#4CAF50"},
                {'range': [40, 70], 'color': "#FFC107"},
                {'range': [70, 100], 'color': "#F44336"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': prob
            }
        },
        number={'font': {'color': "white", 'size': 24}}
    ))
    fig.update_layout(paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Developed by Kiro Medhat | Heart Disease Prediction Project")
