import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# ---------------- CSS Styling ----------------
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #0D0D0D;
    color: #F5F5F5;
}

/* Titles */
h1, h2, h3, h4, h5, h6 {
    color: #FF4B4B;
}

/* Buttons */
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #FF6B6B;
    transform: scale(1.03);
}

/* Inputs */
input, select, textarea {
    background-color: #1C1C1C !important;
    color: #FFFFFF !important;
    border: 1px solid #FF4B4B !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    font-size: 1em !important;
}

/* Placeholder */
::placeholder {
    color: #BBBBBB !important;
}

/* Focus */
input:focus, select:focus, textarea:focus {
    border: 1px solid #FF6B6B !important;
    box-shadow: 0 0 6px #FF4B4B !important;
    outline: none !important;
}

/* Markdown text */
.stMarkdown {
    color: #F5F5F5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("# 🫀 Heart Disease Prediction App")
st.markdown("Enter the patient's details below / أدخل بيانات المريض")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return joblib.load('best_xgboost_model.pkl')

model = load_model()

# ---------------- Input Fields ----------------
st.markdown("### 🧍‍♂️ Patient Information / معلومات المريض")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age (العمر) — Normal: 20–60 yrs", 1, 120, 50)
    sex = st.selectbox("🚻 Sex (0 = Female / أنثى, 1 = Male / ذكر)", [0, 1])
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
                            fbs, restecg, thalch, exang,
                            oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("### 📊 Prediction Result / نتيجة التنبؤ")

    # Risk levels
    if prob < 40:
        st.success(f"🩵 Low Risk ({prob:.2f}%) — Likely healthy / منخفض")
        color = "#10B981"
    elif 40 <= prob <= 70:
        st.warning(f"🟡 Medium Risk ({prob:.2f}%) — May need checkup / متوسط")
        color = "#FBBF24"
    else:
        st.error(f"🔴 High Risk ({prob:.2f}%) — Consult a doctor / عالي")
        color = "#EF4444"

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Heart Disease Risk (%)", 'font': {'color': "#F5F5F5"}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#F5F5F5"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "#10B981"},
                {'range': [40, 70], 'color': "#FBBF24"},
                {'range': [70, 100], 'color': "#EF4444"},
            ]
        }
    ))
    fig.update_layout(paper_bgcolor="#0D0D0D", font_color="#F5F5F5")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Developed by **Kiro Medhat** | ❤️ Heart Disease Prediction Dashboard")
