"""
StreamFit Audit Platform — entry point.
Run with: streamlit run streamlit_app.py

Session state keys:
  page            : 'home' | 'detail' | 'dashboard' | 'recommender'
  selected_id     : interaction_id currently open in Page 1A
  summaries       : dict[interaction_id -> summary str]  (hover cache)
  extracted       : dict[interaction_id -> record dict]  (extraction cache)
  dataframe       : pd.DataFrame | None
  records_cache   : list[dict]
  recommendations : dict | None  (Page 3 LLM output)
"""

import sys
from pathlib import Path

# Make project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="StreamFit Audit Platform",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hide the default Streamlit header decoration */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}

  /* Tile card base */
  .tile-card {
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    padding: 14px 16px;
    background: #fafafa;
    transition: box-shadow 0.2s;
    cursor: pointer;
    min-height: 110px;
  }
  .tile-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.12); }

  /* Churn risk border colours */
  .risk-none    { border-left: 5px solid #2ECC71; }
  .risk-low     { border-left: 5px solid #00CED1; }
  .risk-medium  { border-left: 5px solid #FFA500; }
  .risk-high    { border-left: 5px solid #FF4B4B; }
  .risk-immediate { border-left: 5px solid #8B0000; }
  .risk-unknown { border-left: 5px solid #95A5A6; }

  /* Spec sheet sections */
  .spec-section {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
  }

  /* Score pill */
  .score-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
  }

  /* Channel icon badges */
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state bootstrap ───────────────────────────────────────────────────
defaults = {
    "page": "home",
    "selected_id": None,
    "summaries": {},
    "extracted": {},
    "dataframe": None,
    "records_cache": [],
    "recommendations": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Router ────────────────────────────────────────────────────────────────────
from app.page1_tiles import render_page1
from app.page1a_detail import render_page1a
from app.page2_dashboard import render_page2
from app.page3_recommender import render_page3

page = st.session_state.page

if page == "home":
    render_page1()
elif page == "detail":
    render_page1a()
elif page == "dashboard":
    render_page2()
elif page == "recommender":
    render_page3()
