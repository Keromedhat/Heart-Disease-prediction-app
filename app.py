import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# ---------------- Improved CSS (Dark Gray + Red + Light Inputs) ----------------
st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #181818; /* رمادي غامق مريح */
    color: #FFFFFF;
    font-family: "Segoe UI", sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #FF4B4B;
    font-weight: 700;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #EF4444, #DC2626);
    color: white;
    border: none;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #F87171, #EF4444);
    box-shadow: 0 0 10px #EF4444;
}

/* Input labels */
div[data-baseweb="number-input"] label, 
div[data-baseweb="select"] label {
    color: #E5E5E5 !important;
    font-weight: 600 !important;
}

/* Number Inputs + Select boxes */
div[data-baseweb="number-input"], div[data-baseweb="select"] {
    background-color: #F5F5F5 !important; /* فاتح وواضح */
    color: #000000 !important;
    border-radius: 8px !important;
    border: 1px solid #BBBBBB !important;
    padding: 6px !important;
}
input, select, textarea {
    background-color: #F5F5F5 !important;
    color: #000000 !important;
    border-radius: 8px !important;
}

/* Focus effect */
div[data-baseweb="number-input"]:focus-within,
div[data-baseweb="select"]:focus-within {
    border: 1px solid #EF4444 !important;
    box-shadow: 0 0 6px #EF4444 !important;
}

/* Placeholders */
::placeholder {
    color: #444 !important;
}

/* Markdown text */
.stMarkdown, .stCaption, .stText {
    color: #EDEDED !important;
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
        st.success(f"🩵 Low Risk ({prob:.2f}%) — Lik
