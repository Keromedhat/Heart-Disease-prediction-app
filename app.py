import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# ---------------- CSS for Black + Red Theme ----------------
st.markdown("""
<style>
/* ===== General App Background ===== */
[data-testid="stAppViewContainer"] {
    background-color: #0B0B0B;
    color: #F5F5F5;
}

/* ===== Titles ===== */
h1, h2, h3, h4, h5, h6 {
    color: #EF4444;
}

/* ===== Buttons ===== */
.stButton>button {
    background-color: #EF4444;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
    font-weight: bold;
    border: none;
}

/* ===== Input Fields (fixes visibility issue) ===== */
div[data-baseweb="input"] > div:first-child,
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="number-input"] > div:first-child {
    background-color: #1F1F1F !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: 1px solid #EF4444 !important;
}

/* The actual text input */
input, select {
    background-color: #1F1F1F !important;
    color: #FFFFFF !important;
}

/* Placeholder text */
::placeholder {
    color: #AAAAAA !important;
}

/* Focus highlight */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="number-input"]:focus-within,
div[data-baseweb="select"]:focus-within {
    border: 1px solid #F87171 !important;
    box-shadow: 0 0 6px #EF4444 !important;
}

/* Markdown text */
.stMarkdown {
    color: #F5F5F5;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Title ----------------
st.markdown("## 🫀 Heart Disease Prediction App")
st.markdown("Enter the patient's details below / أدخل بيانات المريض")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load('best_xgboost_model.pkl')
    return model

model = load_model()

# ---------------- Input Fields ----------------
st.markdown("### 🧍‍♂️ Patient Information / معلومات المريض")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age (العمر) — Normal: 20–60 yrs", 1, 120, 50)
    sex = st.selectbox("🚻 Sex (0 = Female / أنثى, 1 = Male / ذكر)", [0, 1])
    cp = st.number_input("💔 Chest Pain Type (0–3) — Normal: 0 = Typical angina / طبيعي", 0, 3, 1)
    trestbps = st.number_input("🩺 Resting Blood Pressure (mmHg) — Normal: 90–120", 80, 200, 120)
    chol = st.number_input("💉 Serum Cholesterol (mg/dl) — Normal: <200", 100, 600, 200)
    fbs = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dl — Normal: 0 = False / طبيعي", [0, 1])

with col2:
    restecg = st.number_input("📈 Resting ECG Results (0–2) — Normal: 0 = Normal / طبيعي", 0, 2, 1)
    thalch = st.number_input("❤️ Max Heart Rate (bpm) — Normal: 120–200", 60, 220, 150)
    exang = st.selectbox("🏃 Exercise Induced Angina — Normal: 0 = No / لا", [0, 1])
    oldpeak = st.number_input("📉 Oldpeak (ST depression) — Normal: 0–1", 0.0, 6.0, 1.0)
    slope = st.number_input("📊 Slope of ST Segment (0–2) — Normal: 1 = Upsloping / طبيعي", 0, 2, 1)
    ca = st.number_input("🫀 Number of Major Vessels (0–3) — Normal: 0", 0, 3, 0)
    thal = st.number_input("🧬 Thalassemia (0=Normal / طبيعي,1=Fixed / ثابت,2=Reversible / عكوس)", 0, 2, 1)

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("### 📊 Prediction Result / نتيجة التنبؤ")

    # Risk levels
    if prob < 40:
        st.success(f"🩵 Low Risk ({prob:.2f}%) — Likely healthy / منخفض")
        color = "#10B981"  # green
    elif 40 <= prob <= 70:
        st.warning(f"🟡 Medium Risk ({prob:.2f}%) — May need checkup / متوسط")
        color = "#FBBF24"  # amber
    else:
        st.error(f"🔴 High Risk ({prob:.2f}%) — Consult a doctor / عالي")
        color = "#EF4444"  # red

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={'reference': 50, 'increasing': {'color': "#EF4444"}},
        title={'text': "Heart Disease Risk (%)", 'font': {'color': "#F5F5F5", 'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#F5F5F5"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "#6EE7B7"},  # light green
                {'range': [40, 70], 'color': "#FCD34D"}, # light amber
                {'range': [70, 100], 'color': "#F87171"} # light red
            ],
            'threshold': {
                'line': {'color': "#F5F5F5", 'width': 4},
                'thickness': 0.75,
                'value': prob
            }
        },
        number={'font': {'color': "#F5F5F5", 'size': 24}}
    ))
    fig.update_layout(paper_bgcolor="#0B0B0B", font_color="#F5F5F5")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Developed by Kiro Medhat | Heart Disease Prediction Dashboard")
