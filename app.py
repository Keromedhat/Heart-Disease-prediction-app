import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- إعداد الصفحة ----------------
st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

# ---------------- العنوان ----------------
st.markdown("## 🫀 Heart Disease Prediction App")
st.markdown("أدخل بيانات المريض بدقة للحصول على النتيجة\nEnter the patient’s details below to get prediction")

# ---------------- تحميل الموديل ----------------
@st.cache_resource
def load_model():
    model = joblib.load('best_xgboost_model.pkl')
    return model

model = load_model()

# ---------------- إدخال البيانات ----------------
st.markdown("### 🧍‍♂️ Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 العمر (Age)", 1, 120, 50)
    st.caption("🔹 Normal range: 20–60 years")

    sex = st.selectbox("🚻 الجنس (Sex)", [0, 1], format_func=lambda x: "أنثى (Female)" if x == 0 else "ذكر (Male)")
    st.caption("🔹 0 = Female | 1 = Male")

    cp = st.selectbox("💔 نوع ألم الصدر (Chest Pain Type)", [0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "🟢 0 = Typical Angina (طبيعي)",
                          1: "🟡 1 = Atypical Angina (مختلف بسيط)",
                          2: "🔴 2 = Non-anginal Pain (غير ذبحة)",
                          3: "🔴 3 = Asymptomatic (بدون أعراض)"
                      }[x])
    st.caption("🔹 Chest pain type classification")

    trestbps = st.number_input("🩺 ضغط الدم وقت الراحة (Resting BP)", 80, 200, 120)
    st.caption("🔹 Normal: 90–120 mmHg")

    chol = st.number_input("💉 الكوليسترول (Cholesterol)", 100, 600, 200)
    st.caption("🔹 Normal: <200 mg/dl")

    fbs = st.selectbox("🩸 سكر الدم الصايم (Fasting Blood Sugar)", [0, 1],
                       format_func=lambda x: "🟢 0 = Normal (<120)" if x == 0 else "🔴 1 = High (>120)")

with col2:
    restecg = st.selectbox("📈 رسم القلب وقت الراحة (Resting ECG)", [0, 1, 2],
                           format_func=lambda x: {
                               0: "🟢 0 = Normal (طبيعي)",
                               1: "🟡 1 = ST-T abnormality (خلل بسيط)",
                               2: "🔴 2 = Left ventricular hypertrophy (تضخم)"
                           }[x])
    st.caption("🔹 ECG result categories")

    thalch = st.number_input("❤️ أقصى معدل نبض (Max Heart Rate)", 60, 220, 150)
    st.caption("🔹 Normal: 120–200 bpm")

    exang = st.selectbox("🏃 ألم مع المجهود (Exercise Angina)", [0, 1],
                         format_func=lambda x: "🟢 0 = No (مافيش)" if x == 0 else "🔴 1 = Yes (فيه)")

    oldpeak = st.number_input("📉 انخفاض ST (Oldpeak)", 0.0, 6.0, 1.0)
    st.caption("🔹 Normal: 0–1")

    slope = st.selectbox("📊 ميل مقطع ST (Slope)", [0, 1, 2],
                         format_func=lambda x: {
                             0: "🔴 0 = Downsloping (نازل)",
                             1: "🟢 1 = Upsloping (طالع طبيعي)",
                             2: "🟡 2 = Flat (مستوٍ)"
                         }[x])
    st.caption("🔹 Shape of ST segment")

    ca = st.selectbox("🫀 عدد الأوعية المسدودة (CA)", [0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "🟢 0 = No blocked vessels (سليم)",
                          1: "🟡 1 = 1 vessel (واحد)",
                          2: "🔴 2 = 2 vessels (اتنين)",
                          3: "🔴 3 = 3 vessels (تلاتة)"
                      }[x])
    st.caption("🔹 Major blood vessels count")

    thal = st.selectbox("🧬 الثلاسيميا (Thalassemia)", [0, 1, 2],
                        format_func=lambda x: {
                            0: "🟢 0 = Normal (طبيعي)",
                            1: "🔴 1 = Fixed Defect (ثابت)",
                            2: "🟡 2 = Reversible (عكوس)"
                        }[x])
    st.caption("🔹 Thalassemia type")

# ---------------- التنبؤ ----------------
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("### 📊 Prediction Result")

    if prob < 40:
        st.success(f"🩵 Low Risk ({prob:.2f}%) — Likely Healthy")
        color = "#10B981"
    elif 40 <= prob <= 70:
        st.warning(f"🟡 Medium Risk ({prob:.2f}%) — Needs Checkup")
        color = "#FBBF24"
    else:
        st.error(f"🔴 High Risk ({prob:.2f}%) — Consult a Doctor")
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

# ---------------- الفوتر ----------------
st.markdown("---")
st.caption("👨‍💻 Developed by Kerolos Medhat")
