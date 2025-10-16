import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- إعداد الصفحة ----------------
st.set_page_config(page_title="تطبيق التنبؤ بأمراض القلب | Heart Disease App", layout="centered")

# ---------------- العنوان ----------------
st.markdown("## 🫀 تطبيق التنبؤ بأمراض القلب | Heart Disease Prediction App")
st.markdown("أدخل بيانات المريض بدقة للحصول على النتيجة\nEnter the patient’s details below to get prediction")

# ---------------- تحميل الموديل ----------------
@st.cache_resource
def load_model():
    model = joblib.load('best_xgboost_model.pkl')
    return model

model = load_model()

# ---------------- إدخال البيانات ----------------
st.markdown("### 🧍‍♂️ بيانات المريض | Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 العمر (Age)", 1, 120, 50)
    st.caption("🔹 الطبيعي من ٢٠ لـ ٦٠ سنة | Normal range: 20–60 years")

    sex = st.selectbox("🚻 الجنس (Sex)", [0, 1], format_func=lambda x: "أنثى (Female)" if x == 0 else "ذكر (Male)")
    st.caption("🔹 0 = أنثى / Female | 1 = ذكر / Male")

    cp = st.number_input("💔 نوع ألم الصدر (Chest Pain Type)", 0, 3, 1)
    st.caption("🔹 0 = طبيعي / Typical angina | 1–3 = أنواع تانية / Other types")

    trestbps = st.number_input("🩺 ضغط الدم وقت الراحة (Resting Blood Pressure)", 80, 200, 120)
    st.caption("🔹 الطبيعي من ٩٠ لـ ١٢٠ | Normal: 90–120 mmHg")

    chol = st.number_input("💉 الكوليسترول (Cholesterol)", 100, 600, 200)
    st.caption("🔹 الطبيعي أقل من ٢٠٠ | Normal: <200 mg/dl")

    fbs = st.selectbox("🩸 سكر الدم الصايم (Fasting Blood Sugar)", [0, 1], format_func=lambda x: "طبيعي (Normal)" if x == 0 else "مرتفع (High)")
    st.caption("🔹 طبيعي أقل من ١٢٠ | Normal: <120 mg/dl")

with col2:
    restecg = st.number_input("📈 رسم القلب وقت الراحة (Resting ECG)", 0, 2, 1)
    st.caption("🔹 0 = طبيعي / Normal | 1–2 = غير طبيعي / Abnormal")

    thalch = st.number_input("❤️ أقصى معدل نبض (Max Heart Rate)", 60, 220, 150)
    st.caption("🔹 الطبيعي من ١٢٠ لـ ٢٠٠ | Normal: 120–200 bpm")

    exang = st.selectbox("🏃 ألم مع المجهود (Exercise Induced Angina)", [0, 1], format_func=lambda x: "لا (No)" if x == 0 else "نعم (Yes)")
    st.caption("🔹 طبيعي: لا | Normal: No")

    oldpeak = st.number_input("📉 انخفاض ST (Oldpeak)", 0.0, 6.0, 1.0)
    st.caption("🔹 الطبيعي من ٠ لـ ١ | Normal: 0–1")

    slope = st.number_input("📊 ميل مقطع ST (Slope)", 0, 2, 1)
    st.caption("🔹 1 = طبيعي / Normal | 0 أو 2 = غير طبيعي / Abnormal")

    ca = st.number_input("🫀 عدد الأوعية المسدودة (CA)", 0, 3, 0)
    st.caption("🔹 الطبيعي = ٠ | Normal: 0")

    thal = st.number_input("🧬 الثلاسيميا (Thalassemia)", 0, 2, 1)
    st.caption("🔹 0 = طبيعي / Normal | 1 = ثابت / Fixed | 2 = عكوس / Reversible")

# ---------------- التنبؤ ----------------
if st.button("🔍 تنبؤ / Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("### 📊 النتيجة / Prediction Result")

    # مستويات الخطر
    if prob < 40:
        st.success(f"🩵 خطر منخفض ({prob:.2f}%) — الحالة مطمئنة / Low Risk — Likely Healthy")
        color = "#10B981"  # green
    elif 40 <= prob <= 70:
        st.warning(f"🟡 خطر متوسط ({prob:.2f}%) — يفضل فحص طبي / Medium Risk — Needs Checkup")
        color = "#FBBF24"  # amber
    else:
        st.error(f"🔴 خطر عالي ({prob:.2f}%) — ضروري تراجع دكتور / High Risk — Consult a Doctor")
        color = "#EF4444"  # red

    # عداد الخطر
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={'reference': 50, 'increasing': {'color': "#EF4444"}},
        title={'text': "نسبة خطر أمراض القلب (%) / Heart Disease Risk (%)", 'font': {'size': 18}},
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
