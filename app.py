import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="VentureIQ · Startup Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. ULTRA-CLEAN DESIGN UI ENGINE (CSS)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    /* Global Background & Typography */
    html, body, .stApp {
        background-color: #04060f !important;
        font-family: 'Inter', sans-serif !important;
        color: #e2e8f0 !important;
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display: none; }

    /* Big Centered Hero Header Bar */
    .hero-container {
        text-align: center;
        padding: 50px 20px 35px 20px;
        background: radial-gradient(circle at top center, rgba(124,58,237,0.18), transparent 65%);
        border-bottom: 1px solid #1e293b;
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -2px;
        color: #ffffff;
        margin-bottom: 12px;
        line-height: 1.1;
    }
    .hero-title span {
        color: #7c3aed;
        text-shadow: 0 0 50px rgba(124,58,237,0.6);
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        color: #94a3b8;
        letter-spacing: -0.2px;
    }
    
    /* Input Form Container (Left) */
    .form-container {
        background: #0b0f19;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 28px;
    }
    
    /* Input Labels Styling */
    div[data-testid="stWidgetLabel"] p, label {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Prediction Result Banner (Right) */
    .result-banner {
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        border: 1px solid;
    }
    .banner-badge {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .banner-title {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 10px;
    }
    .banner-desc {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #94a3b8;
    }

    /* Scoreboard Metrics Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #0b0f19;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
    }
    .metric-box-title {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-box-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Action Plan Strategy Blocks */
    .playbook-container {
        background: #0b0f19;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
    }
    .playbook-row {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        padding: 14px 0;
        border-bottom: 1px solid #1e293b;
    }
    .playbook-row:last-child { border-bottom: none; }
    .playbook-num {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .playbook-text {
        font-size: 0.9rem;
        color: #e2e8f0;
        line-height: 1.5;
    }

    /* Welcome / Idle State Layout */
    .waiting-screen {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 80px 20px;
        text-align: center;
    }
    .waiting-icon {
        font-size: 2.5rem;
        margin-bottom: 16px;
        background: rgba(124, 58, 237, 0.1);
        width: 80px;
        height: 80px;
        display: grid;
        place-items: center;
        border-radius: 50%;
        border: 1px solid rgba(124, 58, 237, 0.2);
    }
    .waiting-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .waiting-desc {
        font-size: 0.9rem;
        color: #64748b;
        max-width: 360px;
        line-height: 1.6;
    }

    /* Scan Action Button Styling */
    .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #7c3aed 0%, #4338ca 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.45) !important;
    }
    
    div[data-testid="stHeaderBlock"] { background: transparent !important; }
    div[data-testid="element-container"] iframe { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# 3. INTERFACE TRANSLATIONS MAPS
CLASS_MAP = {0: 'Acquisition', 1: 'Failure', 2: 'IPO'}
COLOR_MAP = {'Acquisition': '#10b981', 'Failure': '#f43f5e', 'IPO': '#7c3aed'}
EMOJI_MAP = {'Acquisition': '🤝', 'Failure': '⚠️', 'IPO': '🚀'}

# 4. MACHINE LEARNING MODEL INITIALIZATION
@st.cache_resource
def load_ml_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f), None
    except FileNotFoundError:
        return None, "The system file 'model.pkl' could not be found."
    except Exception as e:
        return None, str(e)

model, err = load_ml_model()

def build_features(fr, exp, ts, mkt, users, burn, rev, inv, sec, fb):
    return pd.DataFrame([{
        'funding_rounds':           int(fr),
        'founder_experience_years': int(exp),
        'team_size':                int(ts),
        'market_size_billion':      float(mkt),
        'product_traction_users':   int(users),
        'burn_rate_million':        float(burn),
        'revenue_million':          float(rev),
        'investor_type_none':       1 if inv == 'none'     else 0,
        'investor_type_tier1_vc':   1 if inv == 'tier1_vc' else 0,
        'investor_type_tier2_vc':   1 if inv == 'tier2_vc' else 0,
        'sector_Climate':           1 if sec == 'Climate'   else 0,
        'sector_Crypto':            1 if sec == 'Crypto'    else 0,
        'sector_Ecommerce':         1 if sec == 'Ecommerce' else 0,
        'sector_Fintech':           1 if sec == 'Fintech'   else 0,
        'sector_Health':            1 if sec == 'Health'    else 0,
        'sector_SaaS':              1 if sec == 'SaaS'      else 0,
        'founder_background_ex_bigtech':     1 if fb == 'ex_bigtech'     else 0,
        'founder_background_first_time':     1 if fb == 'first_time'     else 0,
        'founder_background_serial_founder': 1 if fb == 'serial_founder' else 0,
    }])

# 5. RENDER BIG & CENTERED HERO TITLE
st.markdown("""
<div class="hero-container">
  <div class="hero-title">Venture<span>IQ</span> - Startup Predictor</div>
  <div class="hero-subtitle">Advanced Machine Learning Predictive Analysis Engine</div>
</div>
""", unsafe_allow_html=True)

if err:
    st.error(f"Initialization Issue: {err}")
    st.stop()

# 6. TWO-COLUMN BALANCED SYSTEM LAYOUT (With custom side spacing handled inside streamlit columns)
left_outer, center_content, right_outer = st.columns([0.1, 2.5, 0.1])

with center_content:
    left_column, right_column = st.columns([1, 1.4], gap="large")

    # 📊 LEFT COLUMN: CONFIGURATION CONTROL DECK
    with left_column:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="font-size:1.3rem; font-weight:800; margin-bottom:4px; color:#ffffff;">Company Matrix</h2>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.85rem; color:#64748b; margin-bottom:24px;">Adjust metrics to evaluate business trajectory</p>', unsafe_allow_html=True)
        
        funding_rounds = st.number_input("Total Investment Funding Rounds Raised", min_value=0, max_value=20, value=3)
        product_traction_users = st.number_input("Total Active Users Base", min_value=0, max_value=100_000_000, value=50_000, step=10_000)
        founder_experience_years = st.slider("Founder Industry Experience (Years)", min_value=0, max_value=30, value=8)
        
        revenue_million = st.number_input("Annual Recurring Revenue ARR ($ Millions)", min_value=0.0, max_value=2000.0, value=1.5, step=0.5)
        burn_rate_million = st.number_input("Monthly Capital Spending / Burn ($ Millions)", min_value=0.0, max_value=500.0, value=3.0, step=0.5)
        market_size_billion = st.number_input("Total Addressable Market TAM ($ Billions)", min_value=0.1, max_value=5000.0, value=20.0, step=1.0)
        team_size = st.number_input("Total Staff / Employee Headcount", min_value=1, max_value=1000, value=30)
        
        investor_type = st.selectbox("Lead Investor Type", options=['angel', 'none', 'tier2_vc', 'tier1_vc'],
            format_func=lambda x: {'none':'No Investor backing','angel':'Angel Investor Network','tier2_vc':'Tier 2 Venture Capital','tier1_vc':'Tier 1 Venture Capital (Top-tier)'}[x])
        
        sector = st.selectbox("Industry Market Sector", options=['AI', 'Climate', 'Crypto', 'Ecommerce', 'Fintech', 'Health', 'SaaS'])
        founder_background = st.selectbox("Founder Experience Background", options=['academic', 'first_time', 'ex_bigtech', 'serial_founder'],
            format_func=lambda x: {'academic':'Academic / Researcher','first_time':'First-Time Startup Founder','ex_bigtech':'Ex-Big Tech Professional','serial_founder':'Experienced Serial Founder'}[x])
        
        predict_trigger = st.button("⚡ Run Diagnostic Scan")
        st.markdown('</div>', unsafe_allow_html=True)

    # 🔬 RIGHT COLUMN: SIMPLIFIED RESULTS SCREEN
    with right_column:
        if not predict_trigger:
            st.markdown("""
            <div class="waiting-screen">
                <div class="waiting-icon">🔬</div>
                <div class="waiting-title">Ready for System Scan</div>
                <div class="waiting-desc">Fill out the company statistics on the left side panel and click the diagnostic run button to see outcomes.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing Company Health..."):
                input_matrix = build_features(
                    funding_rounds, founder_experience_years, team_size,
                    market_size_billion, product_traction_users,
                    burn_rate_million, revenue_million,
                    investor_type, sector, founder_background
                )
                
                prediction_index = model.predict(input_matrix)[0]
                probability_array = model.predict_proba(input_matrix)[0]
                
                assigned_class = CLASS_MAP[int(prediction_index)]
                probability_mapping = {CLASS_MAP[i]: float(probability_array[i]) for i in range(3)}
                
                prediction_confidence = probability_mapping[assigned_class] * 100
                brand_color = COLOR_MAP[assigned_class]
                class_emoji = EMOJI_MAP[assigned_class]
                
                CONTENT_PLAYBOOKS = {
                    'IPO': {
                        'label': 'High Growth Trajectory',
                        'head': 'Likely heading toward an IPO (Public Listing)',
                        'desc': 'This business demonstrates strong scaling metrics, high user volumes, and healthy capital access. It matches structural properties typical of companies preparing for public stock exchange options.',
                        'steps': ['Begin professional financial structural adjustments to meet future public compliance audits.', 'Establish formal operational oversight councils.', 'Keep accelerating active market adoption numbers to back large value targets.']
                    },
                    'Acquisition': {
                        'label': 'Strategic Value Trajectory',
                        'head': 'Strong Candidate for Strategic Buyout',
                        'desc': 'The model suggests high market worth that makes this business an incredibly attractive purchase target for larger corporate buyers seeking to integrate this team and technology.',
                        'steps': ['Identify top Tier-1 enterprise buyers in the marketplace and form organic integrations.', 'Organize all core technology intellectual blueprints for straightforward valuation processes.', 'Shape product positions to act as an indispensable market plug-in for legacy corporate platforms.']
                    },
                    'Failure': {
                        'label': 'High-Risk Operational Trajectory',
                        'head': 'Financial Runway Warning Flagged',
                        'desc': 'Current operational metrics show rapid capital spending outstripping growth vectors. The model flags a risk of closure or run-out unless immediate steps are taken to balance budgets.',
                        'steps': ['Reduce secondary operational cash burn outlays immediately within 30 days.', 'Refocus core development tasks onto immediate income-producing channels.', 'Set up immediate short-term asset funding lines to maintain balance boundaries.']
                    }
                }
                ActivePlay = CONTENT_PLAYBOOKS[assigned_class]
                
                # A. Main Clean Result Header Card Block
                st.markdown(f"""
                <div class="result-banner" style="background: linear-gradient(135deg, {brand_color}15, transparent); border-color: {brand_color}40;">
                    <div class="banner-badge" style="color: {brand_color};">{class_emoji} {ActivePlay['label']}</div>
                    <div class="banner-title" style="color: #ffffff;">{ActivePlay['head']}</div>
                    <div class="banner-desc">{ActivePlay['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # B. Performance Grid
                burn_ratio = burn_rate_million / revenue_million if revenue_million > 0 else 0
                coverage_factor = revenue_million / burn_rate_million if burn_rate_million > 0 else 0
                
                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-box-title">Prediction Confidence</div>
                        <div class="metric-box-value" style="color: {brand_color};">{prediction_confidence:.1f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-box-title">Burn Multiplier</div>
                        <div class="metric-box-value" style="color: #f59e0b;">{burn_ratio:.2f}x</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-box-title">Revenue-To-Burn Cover</div>
                        <div class="metric-box-value" style="color: #06b6d4;">{coverage_factor:.2f}x</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # C. Clean Outcome Distribution Horizontal Chart
                with st.container(border=True):
                    st.markdown('<div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#64748b; margin-bottom:12px;">Probability Assessment Split</div>', unsafe_allow_html=True)
                    
                    bar_chart = go.Figure()
                    target_categories = ['Acquisition', 'Failure', 'IPO']
                    category_values = [probability_mapping[cat] * 100 for cat in target_categories]
                    category_colors = [COLOR_MAP[cat] for cat in target_categories]
                    
                    bar_chart.add_trace(go.Bar(
                        x=category_values,
                        y=target_categories,
                        orientation='h',
                        marker_color=category_colors,
                        text=[f"{v:.1f}%" for v in category_values],
                        textposition='outside',
                        textfont=dict(color='#ffffff', size=11, family='JetBrains Mono')
                    ))
                    
                    bar_chart.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=5, b=5, l=10, r=60),
                        height=160,
                        xaxis=dict(visible=False, range=[0, 120]),
                        yaxis=dict(tickfont=dict(color='#ffffff', size=12)),
                        barmode='stack'
                    )
                    st.plotly_chart(bar_chart, use_container_width=True, config={'displayModeBar': False})
                
                # D. Strategic Clear Action Items Block
                st.markdown('<div class="playbook-container">', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#64748b; margin-bottom:16px;">Recommended Next Business Steps</div>', unsafe_allow_html=True)
                
                for index, step_item in enumerate(ActivePlay['steps']):
                    st.markdown(f"""
                    <div class="playbook-row">
                        <div class="playbook-num" style="color: {brand_color};">0{index + 1}</div>
                        <div class="playbook-text">{step_item}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# 7. VISUALLY GROUNDED GLOBAL FOOTER
st.markdown("""
<div style="text-align: center; color: #475569; font-size: 0.75rem; padding: 50px 0 30px; border-top: 1px solid #1e293b; margin-top: 40px;">
  VentureIQ · Machine Learning Classification Engine · Built by <strong>K Chintu</strong>
</div>
""", unsafe_allow_html=True)