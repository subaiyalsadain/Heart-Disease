import streamlit as st
import pandas as pd
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="❤️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; } 

.block-container {

    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 780px;
}

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 2rem 1.25rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.08); }
}
.hero h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0 0 0.4rem; letter-spacing: -0.5px; }
.hero p  { color: #94a3b8; font-size: 0.95rem; margin: 0; }

.section-card {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 16px;
    padding: 1.5rem 1.75rem 1rem;
    margin-bottom: 1.25rem;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.section-label .dot { width: 6px; height: 6px; border-radius: 50%; background: #6366f1; display: inline-block; }

.metric-row { display: flex; gap: 10px; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 40px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: #64748b;
    flex: 1;
    text-align: center;
    min-width: 110px;
}
.metric-pill strong { color: #1e293b; font-weight: 600; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    height: 3.2rem !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
    cursor: pointer !important;
}

.result-high {
    background: linear-gradient(135deg, #fff1f1 0%, #ffe4e4 100%);
    border: 1.5px solid #fca5a5;
    border-radius: 16px;
    padding: 1.75rem;
    text-align: center;
    animation: slideIn 0.4s ease;
}
.result-low {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1.5px solid #86efac;
    border-radius: 16px;
    padding: 1.75rem;
    text-align: center;
    animation: slideIn 0.4s ease;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-icon   { font-size: 3rem; }
.result-title  { font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0 0.3rem; }
.result-subtitle { font-size: 0.9rem; color: #64748b; margin: 0; }
.risk-high { color: #dc2626; }
.risk-low  { color: #16a34a; }

.prob-bar-wrap { margin: 1.2rem 0 0.3rem; }
.prob-label { font-size: 0.8rem; color: #64748b; display: flex; justify-content: space-between; margin-bottom: 4px; }
.prob-bar { height: 8px; border-radius: 99px; background: #e2e8f0; overflow: hidden; }
.prob-fill-high { height: 100%; background: linear-gradient(90deg, #f87171, #dc2626); border-radius: 99px; }
.prob-fill-low  { height: 100%; background: linear-gradient(90deg, #4ade80, #16a34a); border-radius: 99px; }

.tip-box {
    background: #f0f9ff;
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #0c4a6e;
    margin-top: 1rem;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e8ecf0;
}
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)
    model   = joblib.load(os.path.join(base, "KNN_heart.pkl"))
    scaler  = joblib.load(os.path.join(base, "scaler.pkl"))
    columns = joblib.load(os.path.join(base, "columns.pkl"))
    return model, scaler, columns

model, scaler, expected_columns = load_assets()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">❤️</span>
    <h1>Heart Risk Predictor</h1>
    <p>Enter your clinical details below — our KNN model will estimate your heart disease risk in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ── Section 1 – Personal ──────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label"><span class="dot"></span> Personal information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 45)
with col2:
    sex = st.selectbox("Biological sex", ["Male (M)", "Female (F)"])
sex_code = "M" if sex.startswith("Male") else "F"
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2 – Vitals ────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label"><span class="dot"></span> Vitals & lab results</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    resting_bp  = st.number_input("Resting blood pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 603, 200)
with col2:
    max_hr     = st.slider("Max heart rate achieved", 60, 220, 150)
    fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL?", [0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No")
st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3 – Cardiac indicators ───────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label"><span class="dot"></span> Cardiac indicators</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox("Chest pain type", ["ATA", "NAP", "TA", "ASY"],
        help="ATA=Atypical Angina  NAP=Non-Anginal Pain  TA=Typical Angina  ASY=Asymptomatic")
    resting_ecg = st.selectbox("Resting ECG result", ["Normal", "ST", "LVH"])
with col2:
    exercise_angina = st.selectbox("Exercise-induced angina?", ["No (N)", "Yes (Y)"])
    st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])
oldpeak = st.slider("Oldpeak — ST depression (exercise vs rest)", 0.0, 6.0, 1.0, 0.1)
angina_code = "Y" if exercise_angina.startswith("Yes") else "N"
st.markdown("""
<div class="tip-box">
    💡 <strong>Oldpeak</strong> measures ST segment depression induced by exercise relative to rest.
    Higher values often indicate more significant coronary artery disease.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Live summary pills ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
    <div class="metric-pill">Age <strong>{age} yrs</strong></div>
    <div class="metric-pill">BP <strong>{resting_bp} mmHg</strong></div>
    <div class="metric-pill">Cholesterol <strong>{cholesterol}</strong></div>
    <div class="metric-pill">Max HR <strong>{max_hr} bpm</strong></div>
    <div class="metric-pill">Oldpeak <strong>{oldpeak:.1f}</strong></div>
</div>
""", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🫀  Predict my heart risk"):

    # Scale numerical features (scaler was fit on 5 cols including Cholesterol)
    num_df = pd.DataFrame(
        [[age, resting_bp, cholesterol, max_hr, oldpeak]],
        columns=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    )
    scaled_num = scaler.transform(num_df)
    scaled_num_df = pd.DataFrame(scaled_num, columns=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'])

    # Binary / categorical features
    bin_input = {
        'FastingBS':                    fasting_bs,
        f'Sex_{sex_code}':              1,
        f'ChestPainType_{chest_pain}':  1,
        f'RestingECG_{resting_ecg}':    1,
        f'ExerciseAngina_{angina_code}':1,
        f'ST_Slope_{st_slope}':         1,
    }
    bin_df = pd.DataFrame([bin_input])

    # Merge, fill missing cols, reorder
    full_df = pd.concat([scaled_num_df, bin_df], axis=1)
    for col in expected_columns:
        if col not in full_df.columns:
            full_df[col] = 0
    full_df = full_df[expected_columns]

    prediction = model.predict(full_df)[0]
    proba      = model.predict_proba(full_df)[0]
    high_pct   = int(round(proba[1] * 100))
    low_pct    = int(round(proba[0] * 100))

    st.markdown("---")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-high">
            <div class="result-icon">⚠️</div>
            <div class="result-title risk-high">High Risk of Heart Disease</div>
            <p class="result-subtitle">
                Your profile shows indicators associated with elevated heart disease risk.<br>
                Please consult a cardiologist or your healthcare provider soon.
            </p>
            <div class="prob-bar-wrap">
                <div class="prob-label"><span>Risk probability</span><span>{high_pct}%</span></div>
                <div class="prob-bar"><div class="prob-fill-high" style="width:{high_pct}%"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box" style="background:#fff7ed; border-color:#fb923c; color:#7c2d12; margin-top:1rem;">
            ❤️ <strong>Next steps:</strong> Schedule a check-up, monitor your blood pressure regularly,
            reduce saturated fats, and ask your doctor about a stress test.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <div class="result-icon">✅</div>
            <div class="result-title risk-low">Low Risk of Heart Disease</div>
            <p class="result-subtitle">
                Your profile shows no major indicators of heart disease risk.<br>
                Keep up your healthy habits and continue routine check-ups.
            </p>
            <div class="prob-bar-wrap">
                <div class="prob-label"><span>Low risk probability</span><span>{low_pct}%</span></div>
                <div class="prob-bar"><div class="prob-fill-low" style="width:{low_pct}%"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box" style="margin-top:1rem;">
            🌿 <strong>Stay healthy:</strong> Maintain regular exercise, a balanced diet, and annual
            cardiovascular screenings — especially as you age.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit · KNN Classifier trained on the UCI Heart Disease Dataset<br>
    <em>This tool is for educational purposes only and does not replace professional medical advice.</em>
</div>
""", unsafe_allow_html=True)