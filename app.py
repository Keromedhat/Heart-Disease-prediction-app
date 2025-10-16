import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(page_title="تطبيق التنبؤ بأمراض القلب", layout="centered")

# ---------------- Title ----------------
st.markdown("## 🫀 تطبيق التنبؤ بأمراض القلب")
st.markdown("أدخل بيانات المريض أدناه بدقة للحصول على التنبؤ")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load('best_xgboost_model.pkl')
    return model

model = load_model()

# ---------------- Input Fields ----------------
st.markdown("### 🧍‍♂️ بيانات المريض")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 العمر (Age)", 1, 120, 50, help="الطبيعي من ٢٠ لـ ٦٠ سنة")
    st.caption("🔹 الطبيعي: من ٢٠ لـ ٦٠ سنة")

    sex = st.selectbox("🚻 الجنس (Sex)", [0, 1], format_func=lambda x: "أنثى" if x == 0 else "ذكر")
    st.caption("🔹 0 = أنثى / 1 = ذكر")

    cp = st.number_input("💔 نوع ألم الصدر (Chest Pain Type)", 0, 3, 1)
    st.caption("🔹 0 = طبيعي / 1–3 = أنواع مختلفة من الألم")

    trestbps = st.number_input("🩺 ضغط الدم أثناء الراحة (Resting BP)", 80, 200, 120)
    st.caption("🔹 الطبيعي من ٩٠ لـ ١٢٠ ملم زئبق")

    chol = st.number_input("💉 الكوليسترول (Cholesterol)", 100, 600, 200)
    st.caption("🔹 الطبيعي أقل من ٢٠٠ مجم/دل")

    fbs = st.selectbox("🩸 سكر الدم الصائم (Fasting Blood Sugar)", [0, 1], format_func=lambda x: "طبيعي (0)" if x == 0 else "مرتفع (1)")
    st.caption("🔹 طبيعي أقل من ١٢٠ مجم/دل")

with col2:
    restecg = st.number_input("📈 نتائج تخطيط القلب (Resting ECG)", 0, 2, 1)
    st.caption("🔹 0 = طبيعي / 1 أو 2 = غير طبيعي")

    thalch = st.number_input("❤️ أقصى معدل نبض (Max Heart Rate)", 60, 220, 150)
    st.caption("🔹 الطبيعي من ١٢٠ لـ ٢٠٠ نبضة بالدقيقة")

    exang = st.selectbox("🏃 ألم مع المجهود (Exercise Angina)", [0, 1], format_func=lambda x: "لا (0)" if x == 0 else "نعم (1)")
    st.caption("🔹 طبيعي: لا يوجد ألم أثناء المجهود")

    oldpeak = st.number_input("📉 انخفاض ST (Oldpeak)", 0.0, 6.0, 1.0)
    st.caption("🔹 الطبيعي من ٠ لـ ١")

    slope = st.number_input("📊 ميل قطعة ST (Slope)", 0, 2, 1)
    st.caption("🔹 1 = طبيعي / 0 أو 2 = غير طبيعي")

    ca = st.number_input("🫀 عدد الأوعية الرئيسية المسدودة (CA)", 0, 3, 0)
    st.caption("🔹 الطبيعي = ٠")

    thal = st.number_input("🧬 الثلاسيميا (Thal)", 0, 2, 1)
    st.caption("🔹 0 = طبيعي / 1 = ثابت / 2 = عكوس")

# ---------------- Prediction ----------------
if st.button("🔍 تنبؤ"):
    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.markdown("### 📊 نتيجة التنبؤ")

    # Risk levels
    if prob < 40:
        st.success(f"🩵 خطر منخفض ({prob:.2f}%) — الحالة مطمئنة")
        color = "#10B981"  # green
    elif 40 <= prob <= 70:
        st.warning(f"🟡 خطر متوسط ({prob:.2f}%) — يُنصح بالفحص الطبي")
        color = "#FBBF24"  # amber
    else:
        st.error(f"🔴 خطر مرتفع ({prob:.2f}%) — يُنصح بمراجعة الطبيب فورًا")
        color = "#EF4444"  # red

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={'reference': 50, 'increasing': {'color': "#EF4444"}},
        title={'text': "نسبة خطر الإصابة بالقلب (%)", 'font': {'size': 20}},
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
st.caption("👨‍💻 تم التطوير بواسطة كيرلس مدحت | تطبيق التنبؤ بأمراض القلب")
