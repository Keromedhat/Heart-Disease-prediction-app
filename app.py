import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

# ---------------- Title ----------------
st.markdown("## 🫀 Heart Disease Prediction App")
st.markdown("Enter the patient’s details below to get prediction")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load('best_xgboost_model.pkl')
    return model

model = load_model()

# ---------------- Input Section ----------------
st.markdown("### 🧍‍♂️ Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", 1, 120, 50)
    st.caption("Normal range: 20–60 years")

    sex = st.selectbox("🚻 Gender", [0, 1],
                       format_func=lambda x: "أنثى (Female)" if x == 0 else "ذكر (Male)")

    cp = st.selectbox("💔 Chest Pain Type", [0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "🟦 S = Typical Angina (طبيعي)",
                          1: "🟩 A = Atypical Angina (مختلف بسيط)",
                          2: "🟧 B = Non-anginal Pain (وجع غير قلبي)",
                          3: "🟥 C = Asymptomatic (مفيش أعراض)"
                      }[x])
    st.caption("Chest pain category from S (Best) to C (Worst)")

    trestbps = st.number_input("🩺 Resting Blood Pressure", 80, 200, 120)
    st.caption("Normal: 90–120 mmHg")

    chol = st.number_input("💉 Serum Cholesterol", 100, 600, 200)
    st.caption("Normal: <200 mg/dl")

    fbs = st.selectbox("🩸 Fasting Blood Sugar", [0, 1],
                       format_func=lambda x: "🟦 S = Normal (<120) طبيعي" if x == 0 else "🟥 C = High (>120) عالي")

with col2:
    restecg = st.selectbox("📈 Resting ECG", [0, 1, 2],
                           format_func=lambda x: {
                               0: "🟦 S = Normal (طبيعي)",
                               1: "🟧 B = ST-T Abnormality (خلل بسيط)",
                               2: "🟥 C = LVH (تضخم في القلب)"
                           }[x])
    st.caption("ECG results from S (Normal) to C (Abnormal)")

    thalch = st.number_input("❤️ Max Heart Rate", 60, 220, 150)
    st.caption("Normal: 120–200 bpm")

    exang = st.selectbox("🏃 Exercise Induced Angina", [0, 1],
                         format_func=lambda x: "🟦 S = No (مافيش)" if x == 0 else "🟥 C = Yes (فيه)")

    oldpeak = st.number_input("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st.caption("Normal: 0–1")

    slope = st.selectbox("📊 Slope of ST Segment", [0, 1, 2],
                         format_func=lambda x: {
                             0: "🟥 C = Downsloping (نازل)",
                             1: "🟦 S = Upsloping (طالع طبيعي)",
                             2: "🟧 B = Flat (مستوٍ)"
                         }[x])
    st.caption("Shape of ST segment from S (Good) to C (Weak)")

    ca = st.selectbox("🫀 Major Vessels (CA)", [0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "🟦 S = 0 (سليم)",
                          1: "🟩 A = 1 (واحد)",
                          2: "🟧 B = 2 (اتنين)",
                          3: "🟥 C = 3 (تلاتة)"
                      }[x])
    st.caption("Blocked vessels count — lower is better")

    thal = st.selectbox("🧬 Thalassemia", [0, 1, 2],
                        format_func=lambda x: {
                            0: "🟦 S = Normal (طبيعي)",
                            1: "🟥 C = Fixed Defect (ثابت)",
                            2: "🟧 B = Reversible (عكوس)"
                        }[x])
    st.caption("Thalassemia levels from S (Normal) to C (Defect)")

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("### 📊 Prediction Result")

    if prob < 40:
        st.success(f"🟩 Low Risk ({prob:.2f}%) — Likely Healthy")
        color = "#10B981"
    elif 40 <= prob <= 70:
        st.warning(f"🟧 Medium Risk ({prob:.2f}%) — Needs Checkup")
        color = "#FBBF24"
    else:
        st.error(f"🟥 High Risk ({prob:.2f}%) — Consult a Doctor")
        color = "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={'reference': 50, 'increasing': {'color': "#EF4444"}},
        title={'text': "Heart Disease Risk (%)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "#6EE7B7"},
                {'range': [40, 70], 'color': "#FCD34D"},
                {'range': [70, 100], 'color': "#F87171"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prob}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("👨‍💻 Developed by Kerolos Medhat")
