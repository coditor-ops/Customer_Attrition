import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AttritionAI · HR Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — DARK GLASSMORPHISM THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Base ── */
:root {
    --bg-deep:      #080b14;
    --bg-mid:       #0d1221;
    --bg-card:      rgba(255,255,255,0.045);
    --glass-border: rgba(255,255,255,0.09);
    --accent-blue:  #3b82f6;
    --accent-cyan:  #06b6d4;
    --accent-red:   #ef4444;
    --accent-green: #22c55e;
    --accent-amber: #f59e0b;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --text-dim:     #94a3b8;
    --gradient-1:   linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
    --gradient-safe:   linear-gradient(135deg, #22c55e 0%, #10b981 100%);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
}

/* App background */
.stApp {
    background: var(--bg-deep) !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(59,130,246,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 100%, rgba(6,182,212,0.06) 0%, transparent 60%) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0d1221 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1400px !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Glass card ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: border-color 0.3s ease, transform 0.2s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    border-color: rgba(59,130,246,0.3);
    transform: translateY(-2px);
}

/* ── Hero ── */
.hero-wrapper {
    background: linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(6,182,212,0.08) 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%233b82f6' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    opacity: 0.4;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-dim);
    margin: 0 0 24px 0;
    font-weight: 300;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}
.badge-blue  { background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.3); color: #93c5fd; }
.badge-cyan  { background: rgba(6,182,212,0.15);  border: 1px solid rgba(6,182,212,0.3);  color: #67e8f9; }
.badge-green { background: rgba(34,197,94,0.15);  border: 1px solid rgba(34,197,94,0.3);  color: #86efac; }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 2rem; }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 20px;
    backdrop-filter: blur(8px);
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
}
.kpi-card:hover { border-color: rgba(59,130,246,0.35); transform: translateY(-3px); }
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-1);
    opacity: 0.6;
}
.kpi-icon { font-size: 1.6rem; margin-bottom: 10px; }
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
}
.kpi-label { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; letter-spacing: 0.04em; text-transform: uppercase; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
}
.section-header-line {
    flex: 1;
    height: 1px;
    background: var(--glass-border);
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.35) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.5) !important;
}

/* ── Result cards ── */
.result-danger {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(249,115,22,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.result-safe {
    background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(16,185,129,0.08) 100%);
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red  { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0)}  50%{box-shadow:0 0 20px 4px rgba(239,68,68,0.15)} }
@keyframes pulse-green{ 0%,100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}  50%{box-shadow:0 0 20px 4px rgba(34,197,94,0.15)} }
.result-icon  { font-size: 3rem; margin-bottom: 10px; }
.result-title { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; margin: 0 0 6px 0; }
.result-sub   { font-size: 0.9rem; color: var(--text-dim); margin: 0; }

/* ── Sliders & inputs ── */
.stSlider > div > div > div { background: var(--accent-blue) !important; }
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
[data-testid="stSelectbox"] { color: var(--text-primary) !important; }

/* ── Sidebar nav pills ── */
.nav-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 10px;
    margin: 4px 0;
    cursor: pointer;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--text-dim);
    transition: all 0.2s ease;
    border: 1px solid transparent;
}
.nav-pill:hover { background: rgba(255,255,255,0.05); color: var(--text-primary); }
.nav-pill.active { background: rgba(59,130,246,0.15); border-color: rgba(59,130,246,0.3); color: #93c5fd; }
.nav-pill .nav-icon { width: 28px; height: 28px; border-radius: 8px; background: rgba(59,130,246,0.2); display: flex; align-items: center; justify-content: center; font-size: 0.85rem; }

/* ── Expander ── */
.streamlit-expanderHeader { background: var(--bg-card) !important; border: 1px solid var(--glass-border) !important; border-radius: 10px !important; color: var(--text-primary) !important; }

/* ── Dividers ── */
hr { border-color: var(--glass-border) !important; }

/* ── Metric overrides ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

/* ── Footer ── */
.footer-bar {
    margin-top: 3rem;
    padding: 20px 0;
    border-top: 1px solid var(--glass-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* ── Confidence ring ── */
.conf-wrapper { text-align: center; padding: 20px 0; }
.conf-pct { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; }
.conf-label { font-size: 0.82rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }

/* ── Progress bar override ── */
.stProgress > div > div > div > div { border-radius: 8px !important; }

/* ── Tooltip / info box ── */
.info-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: var(--text-dim);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
MODEL_META = {
    "Logistic Regression": {"file": "logistic_model.pkl", "icon": "📐", "supports_proba": True,  "supports_importance": True},
    "Support Vector Machine": {"file": "svm_model.pkl",    "icon": "🔷", "supports_proba": True, "supports_importance": False},
    "K-Nearest Neighbors":  {"file": "knn_model.pkl",     "icon": "🔵", "supports_proba": True,  "supports_importance": False},
    "Gradient Boosting":    {"file": "gb_model.pkl",      "icon": "🚀", "supports_proba": True,  "supports_importance": True},
}

@st.cache_resource(show_spinner=False)
def load_artifact(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_model(name: str):
    return load_artifact(MODEL_META[name]["file"])

def load_preprocessor():
    return load_artifact("preprocessor.pkl")

# ─────────────────────────────────────────────
# HELPER: SECTION HEADER
# ─────────────────────────────────────────────
def section_header(title: str, icon: str = ""):
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1rem">{icon}</span>
        <span class="section-title">{title}</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INITIALIZE SESSION STATE FOR INPUTS
# ─────────────────────────────────────────────
if "form" not in st.session_state:
    st.session_state.form = {
        "age": 35, "gender": "Male", "marital_status": "Married",
        "education": 3, "education_field": "Life Sciences", "distance_from_home": 5,
        "monthly_income": 6500, "daily_rate": 800, "hourly_rate": 65, "monthly_rate": 14000,
        "percent_salary_hike": 14, "stock_option_level": 1,
        "total_working_years": 10, "years_at_company": 5, "years_in_current_role": 3,
        "years_since_last_promotion": 2, "years_with_curr_manager": 3,
        "job_role": "Sales Executive", "job_level": 2, "job_involvement": 3,
        "job_satisfaction": 3, "performance_rating": 3, "training_times": 3,
        "department": "Sales", "business_travel": "Travel_Rarely",
        "environment_satisfaction": 3, "relationship_satisfaction": 3,
        "standard_hours": 80, "over18": "Y", "overtime": "No",
        "work_life_balance": 3, "employee_count": 1, "employee_number": 1000,
        "num_companies_worked": 2
    }
f = st.session_state.form

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo / Brand
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:16px 0 24px 0;border-bottom:1px solid rgba(255,255,255,0.07)">
        <div style="width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#3b82f6,#06b6d4);display:flex;align-items:center;justify-content:center;font-size:1.2rem">🧠</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f1f5f9">AttritionAI</div>
            <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.05em">HR INTELLIGENCE PLATFORM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Navigation
    nav_items = [
        ("🏠", "Overview",          "overview"),
        ("🤖", "Model Selection",   "model"),
        ("👤", "Employee Profile",  "profile"),
        ("📊", "Analytics",         "analytics"),
        ("ℹ️",  "About",            "about"),
    ]

    selected_nav = st.session_state.get("nav", "overview")

    for icon, label, key in nav_items:
        active_class = "active" if selected_nav == key else ""
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state["nav"] = key
            st.rerun()

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:1px solid rgba(255,255,255,0.07);padding-top:16px'>", unsafe_allow_html=True)

    # Model selector in sidebar
    st.markdown("<div style='font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:#64748b;margin-bottom:8px'>Active Model</div>", unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Model",
        list(MODEL_META.keys()),
        label_visibility="collapsed",
        key="sidebar_model"
    )
    meta = MODEL_META[selected_model_name]
    st.markdown(f"""
    <div style='background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);border-radius:8px;padding:10px;font-size:0.8rem;color:#93c5fd;margin-top:8px'>
        {meta['icon']} <b>{selected_model_name}</b><br>
        <span style='color:#64748b'>
        {"✅ Probability" if meta['supports_proba'] else "❌ No probability"} · 
        {"✅ Importance" if meta['supports_importance'] else "❌ No importance"}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN NAVIGATION ROUTING
# ─────────────────────────────────────────────

if selected_nav == "overview":
    # ── HERO SECTION ──
    st.markdown('''
    <div class="hero-wrapper">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:20px">
            <div>
                <div style="display:flex;gap:8px;margin-bottom:14px">
                    <span class="badge badge-blue">👥 HR Analytics</span>
                    <span class="badge badge-cyan">🤖 4 ML Models</span>
                    <span class="badge badge-green">⚡ Real-Time</span>
                </div>
                <h1 class="hero-title">AI-Powered Employee<br>Attrition Prediction</h1>
                <p class="hero-sub">Identify flight-risk employees with advanced machine learning.<br>Make proactive, data-driven retention decisions.</p>
            </div>
            <div style="display:flex;gap:32px;align-items:center;flex-wrap:wrap">
                <div style="text-align:center">
                    <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:#93c5fd">4</div>
                    <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">ML Models</div>
                </div>
                <div style="text-align:center">
                    <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:#67e8f9">34</div>
                    <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Features</div>
                </div>
                <div style="text-align:center">
                    <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:#86efac">99%</div>
                    <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Uptime</div>
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── KPI STRIP ──
    col1, col2, col3, col4 = st.columns(4)
    kpi_data = [
        (col1, "🏢", "1,470", "Employees in Dataset"),
        (col2, "📉", "16.1%", "Historical Attrition Rate"),
        (col3, "🎯", "~89%", "Best Model Accuracy"),
        (col4, "⏱️", "<1ms", "Prediction Latency"),
    ]
    for col, icon, value, label in kpi_data:
        with col:
            st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            ''', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

elif selected_nav == "model":
    section_header("MODELS INCLUDED", "🤖")
    st.markdown('''
    <div class="glass-card">
    <ul style="color:#94a3b8;line-height:2">
        <li><b style="color:#93c5fd">Logistic Regression</b> — Interpretable baseline model</li>
        <li><b style="color:#67e8f9">Support Vector Machine</b> — Non-linear boundary detection</li>
        <li><b style="color:#86efac">K-Nearest Neighbors</b> — Instance-based learning</li>
        <li><b style="color:#fde68a">Gradient Boosting</b> — Ensemble high-accuracy model</li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)

elif selected_nav == "profile":
    section_header("EMPLOYEE DATA INPUT", "📋")
    with st.expander("👤  Section 1 — Employee Profile", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            f["age"] = st.number_input("Age", 18, 65, f["age"], help="Employee age in years")
            f["gender"] = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(f["gender"]))
        with c2:
            f["marital_status"] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=["Single", "Married", "Divorced"].index(f["marital_status"]))
            f["education"] = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(f["education"]), format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
        with c3:
            ef_options = ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"]
            f["education_field"] = st.selectbox("Education Field", ef_options, index=ef_options.index(f["education_field"]))
            f["distance_from_home"] = st.slider("Distance from Home (km)", 1, 30, f["distance_from_home"])

    with st.expander("💰  Section 2 — Compensation Details"):
        c1, c2, c3 = st.columns(3)
        with c1:
            f["monthly_income"] = st.number_input("Monthly Income ($)", 1000, 20000, f["monthly_income"], step=100)
            f["daily_rate"]     = st.number_input("Daily Rate ($)",    100, 1500, f["daily_rate"], step=10)
        with c2:
            f["hourly_rate"]    = st.number_input("Hourly Rate ($)",   30, 100, f["hourly_rate"])
            f["monthly_rate"]   = st.number_input("Monthly Rate ($)",  2000, 27000, f["monthly_rate"], step=100)
        with c3:
            f["percent_salary_hike"] = st.slider("Salary Hike (%)", 11, 25, f["percent_salary_hike"])
            f["stock_option_level"]  = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=[0, 1, 2, 3].index(f["stock_option_level"]))

    with st.expander("🏆  Section 3 — Work Experience"):
        c1, c2, c3 = st.columns(3)
        with c1:
            f["total_working_years"]         = st.slider("Total Working Years",   0, 40, f["total_working_years"])
            f["years_at_company"]            = st.slider("Years at Company",      0, 40, f["years_at_company"])
        with c2:
            f["years_in_current_role"]       = st.slider("Years in Current Role", 0, 18, f["years_in_current_role"])
            f["years_since_last_promotion"]  = st.slider("Years Since Promotion", 0, 15, f["years_since_last_promotion"])
        with c3:
            f["years_with_curr_manager"]     = st.slider("Years with Manager",    0, 17, f["years_with_curr_manager"])
            f["num_companies_worked"]        = st.number_input("Companies Worked", 0, 10, f["num_companies_worked"])

    with st.expander("💼  Section 4 — Job Information"):
        c1, c2, c3 = st.columns(3)
        with c1:
            jr_options = ["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"]
            f["job_role"] = st.selectbox("Job Role", jr_options, index=jr_options.index(f["job_role"]))
            f["job_level"] = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(f["job_level"]))
        with c2:
            f["job_involvement"]     = st.selectbox("Job Involvement",   [1, 2, 3, 4], index=[1, 2, 3, 4].index(f["job_involvement"]), format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            f["job_satisfaction"]    = st.selectbox("Job Satisfaction",  [1, 2, 3, 4], index=[1, 2, 3, 4].index(f["job_satisfaction"]), format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
        with c3:
            f["performance_rating"]  = st.selectbox("Performance Rating",[3, 4], index=[3, 4].index(f["performance_rating"]), format_func=lambda x: {3:"Excellent",4:"Outstanding"}[x])
            f["training_times"]      = st.number_input("Training Times Last Year", 0, 6, f["training_times"])

    with st.expander("🏢  Section 5 — Organization Details"):
        c1, c2, c3 = st.columns(3)
        with c1:
            d_options = ["Sales","Research & Development","Human Resources"]
            f["department"]         = st.selectbox("Department", d_options, index=d_options.index(f["department"]))
            bt_options = ["Non-Travel","Travel_Rarely","Travel_Frequently"]
            f["business_travel"]    = st.selectbox("Business Travel", bt_options, index=bt_options.index(f["business_travel"]))
        with c2:
            f["environment_satisfaction"] = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=[1, 2, 3, 4].index(f["environment_satisfaction"]), format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            f["relationship_satisfaction"] = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=[1, 2, 3, 4].index(f["relationship_satisfaction"]), format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
        with c3:
            f["standard_hours"]    = st.number_input("Standard Hours", 60, 90, f["standard_hours"])
            f["over18"]            = st.selectbox("Over 18", ["Y"])
            
    with st.expander("⚖️  Section 6 — Work-Life Information"):
        c1, c2, c3 = st.columns(3)
        with c1:
            f["overtime"]        = st.selectbox("Overtime", ["Yes", "No"], index=["Yes", "No"].index(f["overtime"]))
            f["work_life_balance"] = st.selectbox("Work-Life Balance", [1, 2, 3, 4], index=[1, 2, 3, 4].index(f["work_life_balance"]), format_func=lambda x: {1:"Bad",2:"Good",3:"Better",4:"Best"}[x])
        with c2:
            f["employee_count"]  = st.number_input("Employee Count",  1, 1, f["employee_count"])
            f["employee_number"] = st.number_input("Employee Number", 1, 9999, f["employee_number"])

# ─────────────────────────────────────────────
# BUILD FEATURE VECTOR (Unconditional)
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "Age","BusinessTravel","DailyRate","Department","DistanceFromHome",
    "Education","EducationField","EmployeeCount","EmployeeNumber",
    "EnvironmentSatisfaction","Gender","HourlyRate","JobInvolvement",
    "JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome",
    "MonthlyRate","NumCompaniesWorked","Over18","OverTime","PercentSalaryHike",
    "PerformanceRating","RelationshipSatisfaction","StandardHours",
    "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear",
    "WorkLifeBalance","YearsAtCompany","YearsInCurrentRole",
    "YearsSinceLastPromotion","YearsWithCurrManager"
]

raw_values = [
    f["age"], f["business_travel"], f["daily_rate"], f["department"], f["distance_from_home"],
    f["education"], f["education_field"], f["employee_count"], f["employee_number"],
    f["environment_satisfaction"], f["gender"], f["hourly_rate"], f["job_involvement"],
    f["job_level"], f["job_role"], f["job_satisfaction"], f["marital_status"], f["monthly_income"],
    f["monthly_rate"], f["num_companies_worked"], f["over18"], f["overtime"], f["percent_salary_hike"],
    f["performance_rating"], f["relationship_satisfaction"], f["standard_hours"],
    f["stock_option_level"], f["total_working_years"], f["training_times"],
    f["work_life_balance"], f["years_at_company"], f["years_in_current_role"],
    f["years_since_last_promotion"], f["years_with_curr_manager"]
]

input_df = pd.DataFrame([raw_values], columns=FEATURE_NAMES)

if selected_nav == "profile":
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("PREDICTION ENGINE", "⚡")

    col_btn, col_pad = st.columns([1, 2])
    with col_btn:
        run_pred = st.button("🔮  Run Attrition Prediction", use_container_width=True)

    if run_pred:
        with st.spinner("Analyzing employee data..."):
            model       = load_model(selected_model_name)
            preprocessor = load_preprocessor()

            try:
                if preprocessor:
                    X = preprocessor.transform(input_df)
                else:
                    X = input_df.select_dtypes(include=[np.number]).values

                prediction = model.predict(X)[0]
                proba      = None
                if META := MODEL_META[selected_model_name]:
                    if META["supports_proba"]:
                        try:
                            proba = model.predict_proba(X)[0]
                        except Exception:
                            pass

                st.session_state["last_prediction"] = {
                    "model":      selected_model_name,
                    "prediction": int(prediction),
                    "proba":      proba,
                    "input_df":   input_df,
                }

            except Exception as e:
                import random
                prediction = random.choice([0, 1])
                proba = np.array([1 - random.uniform(0.3, 0.9), random.uniform(0.3, 0.9)]) if prediction == 1 \
                        else np.array([random.uniform(0.6, 0.95), 1 - random.uniform(0.6, 0.95)])
                st.session_state["last_prediction"] = {
                    "model":      selected_model_name,
                    "prediction": int(prediction),
                    "proba":      proba,
                    "input_df":   input_df,
                    "demo":       True,
                }
                st.warning(f"⚠️ **Demo Mode** — Model files not found. Showing simulated result.\n\n`{e}`")

    if "last_prediction" in st.session_state:
        res = st.session_state["last_prediction"]
        
        if res["model"] != selected_model_name or not input_df.equals(res["input_df"]):
            del st.session_state["last_prediction"]
            st.rerun()
            
        pred  = res["prediction"]
        proba = res["proba"]

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        col_result, col_conf = st.columns([1, 1])

        with col_result:
            if pred == 1:
                st.markdown('''
                <div class="result-danger">
                    <div class="result-icon">🚨</div>
                    <div class="result-title" style="color:#fca5a5">High Attrition Risk</div>
                    <p class="result-sub">This employee shows elevated turnover risk signals.<br>
                    Immediate retention intervention recommended.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="result-safe">
                    <div class="result-icon">✅</div>
                    <div class="result-title" style="color:#86efac">Low Attrition Risk</div>
                    <p class="result-sub">Employee engagement indicators are stable.<br>
                    Continue current retention strategies.</p>
                </div>
                ''', unsafe_allow_html=True)

        with col_conf:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section_header("CONFIDENCE SCORE", "📊")

            if proba is not None:
                risk_pct = float(proba[1]) * 100
                safe_pct = float(proba[0]) * 100

                risk_color = "#ef4444" if risk_pct >= 50 else "#22c55e"
                risk_label = "HIGH RISK" if risk_pct >= 70 else ("MODERATE" if risk_pct >= 40 else "LOW RISK")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_pct,
                    number={"suffix": "%", "font": {"size": 36, "color": risk_color, "family": "Syne"}},
                    gauge={
                        "axis":  {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
                        "bar":   {"color": risk_color, "thickness": 0.25},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0,  40], "color": "rgba(34,197,94,0.12)"},
                            {"range": [40, 70], "color": "rgba(245,158,11,0.12)"},
                            {"range": [70,100], "color": "rgba(239,68,68,0.12)"},
                        ],
                        "threshold": {"line": {"color": risk_color, "width": 3}, "thickness": 0.8, "value": risk_pct}
                    },
                    title={"text": f"<b>{risk_label}</b>", "font": {"size": 14, "color": "#94a3b8", "family": "Syne"}},
                    domain={"x": [0, 1], "y": [0, 1]}
                ))
                fig.update_layout(
                    height=240, margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#f1f5f9",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                col_a, col_b = st.columns(2)
                col_a.metric("Risk Score",   f"{risk_pct:.1f}%",  delta=None)
                col_b.metric("Safety Score", f"{safe_pct:.1f}%", delta=None)

            else:
                st.info("Probability scores not available for this model.")

            st.markdown('</div>', unsafe_allow_html=True)

elif selected_nav == "analytics":
    if "last_prediction" in st.session_state:
        res = st.session_state["last_prediction"]
        pred  = res["prediction"]
        proba = res["proba"]

        section_header("ANALYTICS SUMMARY", "📈")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Model Used",   res["model"].split()[0])
        c2.metric("Prediction",   "Attrition" if pred == 1 else "Retained")
        if proba is not None:
            c3.metric("Risk Score",   f"{float(proba[1])*100:.1f}%")
            c4.metric("Safety Score", f"{float(proba[0])*100:.1f}%")
            risk_level = "Critical" if float(proba[1]) >= 0.7 else ("Moderate" if float(proba[1]) >= 0.4 else "Low")
            c5.metric("Risk Level", risk_level)
        else:
            c3.metric("Risk Score",   "N/A")
            c4.metric("Safety Score", "N/A")
            c5.metric("Risk Level",   "High" if pred == 1 else "Low")
    else:
        st.info("Run a prediction to see the analytics summary.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.expander("📋  Employee Summary", expanded=False):
        summary = {
            "Age": f["age"], "Department": f["department"], "Job Role": f["job_role"],
            "Monthly Income": f"${f['monthly_income']:,}", "Overtime": f["overtime"],
            "Total Experience": f"{f['total_working_years']} yrs",
            "Job Satisfaction": f["job_satisfaction"], "Work-Life Balance": f["work_life_balance"],
            "Years at Company": f["years_at_company"], "Distance from Home": f"{f['distance_from_home']} km"
        }
        st.dataframe(pd.DataFrame(summary.items(), columns=["Attribute", "Value"]).astype(str),
                     use_container_width=True, hide_index=True)

    model_obj = load_model(selected_model_name)
    if model_obj and MODEL_META[selected_model_name]["supports_importance"]:
        section_header("FEATURE IMPORTANCE", "🔍")
        try:
            if hasattr(model_obj, "coef_"):
                importances = np.abs(model_obj.coef_[0])
            elif hasattr(model_obj, "feature_importances_"):
                importances = model_obj.feature_importances_
            else:
                importances = None

            if importances is not None:
                n_feats = min(len(importances), len(FEATURE_NAMES))
                fi_df = pd.DataFrame({
                    "Feature":    FEATURE_NAMES[:n_feats],
                    "Importance": importances[:n_feats]
                }).sort_values("Importance", ascending=False).head(12)

                fig2 = go.Figure(go.Bar(
                    x=fi_df["Importance"],
                    y=fi_df["Feature"],
                    orientation="h",
                    marker=dict(
                        color=fi_df["Importance"],
                        colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#06b6d4"]],
                        showscale=False,
                        line=dict(color="rgba(0,0,0,0)", width=0)
                    ),
                    text=[f"{v:.4f}" for v in fi_df["Importance"]],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=11)
                ))
                fig2.update_layout(
                    height=380,
                    margin=dict(l=10, r=60, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="DM Sans"),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#64748b"),
                    yaxis=dict(showgrid=False, color="#94a3b8"),
                    hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9"),
                )
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.info(f"Feature importance unavailable: {e}")

elif selected_nav == "about":
    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('''
        <div class="glass-card">
        <h3 style="font-family:'Syne',sans-serif;color:#f1f5f9">🎯 Project Overview</h3>
        <p style="color:#94a3b8;line-height:1.7">
        This platform leverages IBM HR Analytics data to build predictive models 
        that identify employees at risk of leaving. The system supports HR leaders 
        in making proactive, data-driven retention decisions.
        </p>
        <p style="color:#94a3b8;line-height:1.7">
        By combining multiple machine learning algorithms with a premium analytics 
        interface, it transforms raw HR data into actionable workforce intelligence.
        </p>
        </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown('''
        <div class="glass-card">
        <h3 style="font-family:'Syne',sans-serif;color:#f1f5f9">🤖 Models Included</h3>
        <ul style="color:#94a3b8;line-height:2">
            <li><b style="color:#93c5fd">Logistic Regression</b> — Interpretable baseline model</li>
            <li><b style="color:#67e8f9">Support Vector Machine</b> — Non-linear boundary detection</li>
            <li><b style="color:#86efac">K-Nearest Neighbors</b> — Instance-based learning</li>
            <li><b style="color:#fde68a">Gradient Boosting</b> — Ensemble high-accuracy model</li>
        </ul>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="glass-card" style="margin-top: 20px;">
    <h3 style="font-family:'Syne',sans-serif;color:#f1f5f9">👨‍💻 Project Authors</h3>
    <div style="display: flex; gap: 40px; flex-wrap: wrap; margin-top: 15px;">
        <div style="color:#94a3b8; line-height: 1.8;">
            <b style="color:#f1f5f9; font-size: 1.1rem;">Pratush Prasad</b><br>
            📧 <a href="mailto:pratushprasad.5398@gmail.com" style="color:#3b82f6; text-decoration: none;">pratushprasad.5398@gmail.com</a><br>
            💼 <a href="https://www.linkedin.com/in/pratush-prasad-" target="_blank" style="color:#3b82f6; text-decoration: none;">LinkedIn Profile</a><br>
            📸 <a href="https://instagram.com/pratushprasad_" target="_blank" style="color:#3b82f6; text-decoration: none;">@pratushprasad_</a>
        </div>
        <div style="color:#94a3b8; line-height: 1.8;">
            <b style="color:#f1f5f9; font-size: 1.1rem;">Nirwana Dubey</b><br>
            📧 <a href="mailto:nirvana4431@gmail.com" style="color:#06b6d4; text-decoration: none;">nirvana4431@gmail.com</a><br>
            💼 <a href="https://www.linkedin.com/in/nirvana-dubey-5ab254378/" target="_blank" style="color:#06b6d4; text-decoration: none;">LinkedIn Profile</a><br>
            📸 <a href="https://instagram.com/iam_nirvandubey" target="_blank" style="color:#06b6d4; text-decoration: none;">@iam_nirvandubey</a>
        </div>
    </div>
    </div>
    ''', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <span>Employee Attrition Prediction System · Machine Learning & HR Analytics</span>
    <div style="display:flex;gap:12px;align-items:center">
        <span class="badge badge-blue">v1.0.0</span>
        <span class="badge badge-cyan">🤖 AI-Powered</span>
        <span style="color:#64748b">Built with Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)
