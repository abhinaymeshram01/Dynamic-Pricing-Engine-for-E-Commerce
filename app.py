import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="💹",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Mono', monospace !important;
    background-color: #08090d !important;
    color: #f0f1f5 !important;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 2rem 4rem !important; max-width: 860px !important; }

/* ── Background grid ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(232,255,71,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(232,255,71,.03) 1px, transparent 1px);
    background-size: 48px 48px;
}

/* ── Header badge ── */
.tag-badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 11px; letter-spacing: .2em; text-transform: uppercase;
    color: #e8ff47;
    background: rgba(232,255,71,.08);
    border: 1px solid rgba(232,255,71,.22);
    padding: 6px 14px; border-radius: 2px;
    margin-bottom: 18px;
}
.dot { display:inline-block; width:6px; height:6px; background:#e8ff47;
       border-radius:50%; margin-right:4px; animation: blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

/* ── Main title ── */
.main-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(36px, 7vw, 72px);
    font-weight: 800; line-height: 1; letter-spacing: -.03em;
    margin: 0 0 6px;
}
.accent { color: #e8ff47; }
.subtitle {
    font-size: 12px; color: #6b6f7e; letter-spacing: .07em; margin-bottom: 44px;
}

/* ── Section label ── */
.section-label {
    font-size: 10px; letter-spacing: .22em; text-transform: uppercase;
    color: #6b6f7e; margin-bottom: 18px; padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,.07);
}

/* ── Field card wrapper ── */
.field-card {
    background: #0f1117;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 4px;
    padding: 22px 24px 18px;
    margin-bottom: 12px;
    transition: border-color .2s;
}
.field-card:hover { border-color: rgba(232,255,71,.18); }

.field-title {
    font-size: 10px; letter-spacing: .18em; text-transform: uppercase;
    color: #6b6f7e; margin-bottom: 6px;
}

/* ── Streamlit widget overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #161820 !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 2px !important;
    color: #f0f1f5 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #e8ff47 !important;
    box-shadow: 0 0 0 3px rgba(232,255,71,.12) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: #161820 !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 2px !important;
    color: #f0f1f5 !important;
    font-family: 'Space Mono', monospace !important;
}

/* radio */
[data-testid="stRadio"] label {
    font-size: 13px !important;
    color: #f0f1f5 !important;
}
[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 12px;
}

/* slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #e8ff47 !important;
    box-shadow: 0 0 10px rgba(232,255,71,.5) !important;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: #e8ff47 !important;
    color: #08090d !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 14px 32px !important;
    margin-top: 24px !important;
    transition: transform .15s, box-shadow .15s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(232,255,71,.3) !important;
    background: #f0ff6a !important;
}

/* ── Success result box ── */
.result-box {
    position: relative;
    background: #0f1117;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 4px;
    padding: 32px 28px 28px;
    margin-top: 32px;
    overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00d4aa, #e8ff47, #ff6b35);
}
.result-label {
    font-size: 10px; letter-spacing: .25em; text-transform: uppercase;
    color: #6b6f7e; margin-bottom: 10px;
}
.result-price {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(48px, 10vw, 80px);
    font-weight: 800; letter-spacing: -.04em; line-height: 1;
    color: #e8ff47;
    text-shadow: 0 0 40px rgba(232,255,71,.35);
}
.result-meta {
    display: flex; flex-wrap: wrap; gap: 24px;
    margin-top: 24px; padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,.07);
}
.meta-item { display: flex; flex-direction: column; gap: 3px; }
.meta-key { font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: #6b6f7e; }
.meta-val { font-size: 14px; font-weight: 700; color: #f0f1f5; }
.meta-val.green { color: #00d4aa; }
.meta-val.orange { color: #ff6b35; }
.meta-val.yellow { color: #e8ff47; }

/* ── Error box ── */
.error-box {
    background: rgba(255,107,53,.08);
    border: 1px solid rgba(255,107,53,.3);
    border-radius: 4px; padding: 16px 20px; margin-top: 20px;
    font-size: 13px; color: #ff6b35;
}

/* ── Footer ── */
.footer {
    margin-top: 52px; padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,.06);
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 10px;
}
.footer-note { font-size: 11px; color: #6b6f7e; line-height: 1.7; }
.model-badge {
    font-size: 11px; color: #6b6f7e;
    background: #0f1117; border: 1px solid rgba(255,255,255,.07);
    padding: 6px 12px; border-radius: 2px;
}

/* divider */
.divider { border: none; border-top: 1px solid rgba(255,255,255,.07); margin: 28px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("pricing_model.pkl")

model = load_model()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tag-badge"><span class="dot"></span>ML-Powered · Real-Time</div>
<div class="main-title">Dynamic<br><span class="accent">Pricing</span> Engine</div>
<div class="subtitle">// Enter product variables to compute the optimal price recommendation</div>
""", unsafe_allow_html=True)


# ── Inputs ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">◎ &nbsp; Input Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="field-title">Competitor Pricing ($)</div>', unsafe_allow_html=True)
    competitor_price = st.number_input(
        "Competitor Pricing ($)", min_value=0.0, max_value=200.0,
        value=50.0, step=0.01, label_visibility="collapsed"
    )

    st.markdown('<div class="field-title" style="margin-top:16px">Demand Forecast</div>', unsafe_allow_html=True)
    demand_forecast = st.number_input(
        "Demand Forecast", min_value=0.0, max_value=500.0,
        value=100.0, step=0.1, label_visibility="collapsed"
    )

    st.markdown('<div class="field-title" style="margin-top:16px">Inventory Level</div>', unsafe_allow_html=True)
    inventory_level = st.number_input(
        "Inventory Level", min_value=0, max_value=1000,
        value=200, step=1, label_visibility="collapsed"
    )

    st.markdown('<div class="field-title" style="margin-top:16px">Discount (%)</div>', unsafe_allow_html=True)
    discount = st.number_input(
        "Discount (%)", min_value=0, max_value=100,
        value=10, step=1, label_visibility="collapsed"
    )

with col2:
    st.markdown('<div class="field-title">Seasonality</div>', unsafe_allow_html=True)
    seasonality = st.selectbox(
        "Seasonality",
        options=["Autumn", "Spring", "Summer", "Winter"],
        index=0, label_visibility="collapsed"
    )

    st.markdown('<div class="field-title" style="margin-top:16px">Holiday / Promotion</div>', unsafe_allow_html=True)
    holiday_promo = st.radio(
        "Holiday / Promotion", options=[0, 1],
        format_func=lambda x: "✓ Active" if x else "✕ None",
        index=0, horizontal=True, label_visibility="collapsed"
    )

    st.markdown('<div class="field-title" style="margin-top:16px">Units Ordered</div>', unsafe_allow_html=True)
    units_ordered = st.number_input(
        "Units Ordered", min_value=0, max_value=500,
        value=50, step=1, label_visibility="collapsed"
    )


# ── Predict button ─────────────────────────────────────────────────────────────
predict = st.button("Compute Optimal Price →", type="primary")

if predict:
    input_data = pd.DataFrame({
        "Competitor Pricing": [competitor_price],
        "Demand Forecast":    [demand_forecast],
        "Inventory Level":    [inventory_level],
        "Discount":           [discount],
        "Seasonality":        [seasonality],
        "Holiday/Promotion":  [holiday_promo],
        "Units Ordered":      [units_ordered],
    })

    try:
        prediction = model.predict(input_data)[0]
        season_icons = {"Autumn": "🍂", "Spring": "🌸", "Summer": "☀️", "Winter": "❄️"}
        icon = season_icons.get(seasonality, "")

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">// Recommended Price</div>
            <div class="result-price">${prediction:,.2f}</div>
            <div class="result-meta">
                <div class="meta-item">
                    <span class="meta-key">Competitor Price</span>
                    <span class="meta-val">${competitor_price:.2f}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-key">Demand Forecast</span>
                    <span class="meta-val">{demand_forecast:.1f} units</span>
                </div>
                <div class="meta-item">
                    <span class="meta-key">Inventory</span>
                    <span class="meta-val">{inventory_level} stock</span>
                </div>
                <div class="meta-item">
                    <span class="meta-key">Seasonality</span>
                    <span class="meta-val green">{icon} {seasonality}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-key">Discount Applied</span>
                    <span class="meta-val orange">{discount}%</span>
                </div>
                <div class="meta-item">
                    <span class="meta-key">Promo Active</span>
                    <span class="meta-val yellow">{"✓ Yes" if holiday_promo else "✕ No"}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-key">Units Ordered</span>
                    <span class="meta-val">{units_ordered} pcs</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            ⚠ Prediction failed: {e}
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span class="footer-note">
        Model trained on retail data<br>
        Features: competitor pricing · demand · inventory · discount · seasonality · promo · units ordered
    </span>
    <span class="model-badge">◈ pricing_model.pkl · ML Pipeline</span>
</div>
""", unsafe_allow_html=True)
