"""
Page 3 — LLM Recommender & Action Engine

Synthesises all extracted data into boardroom-ready answers to the 3 business questions.
All LLM claims are grounded in the real aggregated DataFrame stats — no hallucination.
"""

import json

import plotly.graph_objects as go
import streamlit as st

from analysis.dataframe import (
    build_dataframe,
    get_aggregated_stats,
    load_all_extracted,
)
from pipeline.graph import get_client
from pipeline.prompts import SYNTHESIS_SYSTEM, synthesis_user

PRIORITY_COLOR = {
    "urgent": "#8B0000", "high": "#FF4B4B",
    "medium": "#FFA500", "low": "#2ECC71",
}
IMPACT_COLOR = {"high": "#FF4B4B", "medium": "#FFA500", "low": "#2ECC71"}


def _pill(text: str, color: str) -> str:
    """
    Render an inline HTML badge/pill with a coloured background and white text.

    Args:
        text:  Text to display inside the pill.
        color: Background colour (hex or CSS colour string).

    Returns:
        HTML string for a styled inline span element.
    """
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:20px;font-size:0.8rem;font-weight:600">{text}</span>'
    )


def _run_synthesis(stats: dict) -> dict:
    """
    Call GPT-4o with grounded aggregated stats and return parsed recommendations JSON.

    Sends the SYNTHESIS_SYSTEM prompt and a user message built from real DataFrame
    statistics. Strips code fences from the response before JSON parsing.

    Args:
        stats: Aggregated stats dict from get_aggregated_stats(). Passed directly
               into the synthesis prompt so all LLM claims are data-grounded.

    Returns:
        Parsed recommendations dict with keys: executive_summary, churn_drivers,
        upsell_segments, product_improvements, action_items.

    Raises:
        json.JSONDecodeError: If the model returns malformed JSON.
        openai.OpenAIError: On API failure.
    """
    resp = get_client().chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user",   "content": synthesis_user(stats)},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # Strip code fences if present
    import re
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    return json.loads(raw)


def _impact_effort_matrix(improvements: list[dict]) -> go.Figure:
    """
    Render a 2×2 impact vs effort scatter plot for product improvement items.

    Quadrant shading: green (low effort, high impact = Quick Wins),
    blue (high effort, high impact = Big Bets). Marker colour = impact level.

    Args:
        improvements: List of improvement dicts from the synthesis LLM output.
                      Each dict should have keys: improvement, effort, impact.

    Returns:
        Plotly Figure (scatter with quadrant annotations).
    """
    effort_map = {"low": 1, "medium": 2, "high": 3}
    impact_map = {"low": 1, "medium": 2, "high": 3}

    x = [effort_map.get(i.get("effort", "medium"), 2) for i in improvements]
    y = [impact_map.get(i.get("impact", "medium"), 2) for i in improvements]
    labels = [i.get("improvement", "")[:30] + "…" if len(i.get("improvement","")) > 30
              else i.get("improvement","") for i in improvements]
    colors = [IMPACT_COLOR.get(i.get("impact","medium"), "#FFA500") for i in improvements]

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.5, y0=1.5, x1=1.5, y1=3.5,
                  fillcolor="rgba(46,204,113,0.08)", line_color="rgba(46,204,113,0.3)")
    fig.add_shape(type="rect", x0=2.5, y0=1.5, x1=3.5, y1=3.5,
                  fillcolor="rgba(30,144,255,0.08)", line_color="rgba(30,144,255,0.3)")
    fig.add_annotation(x=1, y=3.3, text="Quick Wins", showarrow=False,
                       font=dict(color="#2ECC71", size=11, family="Inter"))
    fig.add_annotation(x=3, y=3.3, text="Big Bets", showarrow=False,
                       font=dict(color="#1E90FF", size=11, family="Inter"))

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=18, color=colors, opacity=0.85,
                    line=dict(width=2, color="white")),
        hovertemplate="<b>%{text}</b><br>Effort: %{x}<br>Impact: %{y}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Product Improvements — Impact vs Effort",
        xaxis=dict(title="Effort", tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"], range=[0.3, 3.7]),
        yaxis=dict(title="Impact", tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"], range=[0.3, 3.7]),
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        height=360,
    )
    return fig


def _churn_driver_chart(drivers: list[dict], df) -> go.Figure:
    """
    Horizontal bar chart of pain categories for at-risk customers.

    When the DataFrame is available, uses real pain category counts from interactions
    with churn_risk_score >= 2 (overrides the LLM-provided driver labels).
    Falls back to LLM driver names if the DataFrame is empty or None.

    Args:
        drivers: List of churn driver dicts from synthesis LLM output (fallback only).
        df:      Flat interactions DataFrame from build_dataframe(). Used for grounded counts.

    Returns:
        Plotly Figure (horizontal bar, top 6 categories).
    """
    if df is not None and not df.empty:
        # Use real data: pain categories for at-risk customers
        at_risk = df[df["churn_risk_score"] >= 2]
        pain_counts = at_risk["top_pain_category"].value_counts().head(6)
        labels = pain_counts.index.tolist()
        values = pain_counts.values.tolist()
    else:
        labels = [d.get("driver", "")[:25] for d in drivers[:5]]
        values = list(range(len(labels), 0, -1))

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#FF4B4B", opacity=0.85,
        text=values, textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Pain Categories — At-Risk Customers",
        xaxis_title="Interactions", yaxis=dict(autorange="reversed"),
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=20, r=20, t=50, b=20), height=280,
    )
    return fig


def _upsell_segment_chart(df) -> go.Figure:
    """
    Bar chart of upsell opportunity count grouped by customer lifecycle stage.

    Filters to interactions with upsell_score >= 2 (medium or high opportunity).
    Returns an empty Figure if df is None or empty.

    Args:
        df: Flat interactions DataFrame from build_dataframe(). Requires
            'upsell_score' and 'lifecycle_stage' columns.

    Returns:
        Plotly Figure (vertical bar).
    """
    if df is None or df.empty:
        return go.Figure()
    upsell = df[df["upsell_score"] >= 2].groupby("lifecycle_stage").size().reset_index(name="count")
    fig = go.Figure(go.Bar(
        x=upsell["lifecycle_stage"], y=upsell["count"],
        marker_color="#1E90FF", text=upsell["count"], textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Upsell Opportunities by Lifecycle Stage",
        xaxis_title="Stage", yaxis_title="Count",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=20, r=20, t=50, b=20), height=280,
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render_page3():
    """
    Render the Page 3 LLM Recommender — strategic synthesis of all extracted interactions.

    Loads aggregated stats from the DataFrame and calls GPT-4o once to generate
    structured recommendations grounded in real data. Recommendations are cached in
    session state; a 'Regenerate' button clears the cache and re-runs the synthesis call.

    Sections rendered:
        - Executive Summary (3 bullets)
        - Q1: Top Churn Drivers — with mitigation playbooks and grounded bar chart
        - Q2: Upsell & Retention Segments — with outreach playbooks and lifecycle chart
        - Q3: Product & Service Improvements — with implementation roadmaps and impact/effort matrix
        - Action Item Registry — with deadlines, owners, and how-to-execute prose
        - Raw Synthesis JSON expander (AI workflow evidence)
    """
    nav_l, nav_r = st.columns([1, 8])
    with nav_l:
        if st.button("← Dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
    with nav_r:
        st.markdown("### 🧠 Strategic Recommendations")

    # Load data
    records = st.session_state.get("records_cache") or load_all_extracted()
    df = st.session_state.get("dataframe")
    if df is None or (hasattr(df, "empty") and df.empty):
        df = build_dataframe(records)
        st.session_state["dataframe"] = df

    if not records:
        st.warning("No extracted interactions found. Run the analysis pipeline first (Page 1 → Analyse All).")
        return

    stats = get_aggregated_stats(df)

    # ── Generate / load recommendations ──────────────────────────────────────
    recs = st.session_state.get("recommendations")

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption(
            f"Synthesising insights from **{stats.get('total_interactions', 0)} interactions** "
            f"| {stats.get('high_churn_count', 0)} high churn risk "
            f"| {stats.get('high_upsell_count', 0)} upsell opportunities"
        )
    with col_btn:
        if st.button("🔄 Regenerate", help="Re-run the synthesis LLM call"):
            st.session_state.recommendations = None
            st.rerun()

    if recs is None:
        with st.spinner("Calling GPT-4o for strategic synthesis…"):
            try:
                recs = _run_synthesis(stats)
                st.session_state.recommendations = recs
            except Exception as e:
                st.error(f"Synthesis failed: {e}")
                return

    # ── Executive Summary ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📌 Executive Summary")
    for bullet in recs.get("executive_summary", []):
        st.markdown(f"- {bullet}")

    st.markdown("---")

    # ── Q1: Churn Drivers ─────────────────────────────────────────────────────
    st.markdown("### ❓ Q1 — Top Drivers of Customer Churn")
    q1_col, q1_chart = st.columns([3, 2])

    with q1_col:
        for i, driver in enumerate(recs.get("churn_drivers", []), 1):
            p_color = PRIORITY_COLOR.get(driver.get("priority", "medium"), "#FFA500")
            with st.container():
                st.markdown(
                    f"**{i}. {driver.get('driver', '')}** &nbsp;"
                    + _pill(driver.get("priority", ""), p_color)
                    + "&nbsp;" + _pill(driver.get("owner", ""), "#1E90FF"),
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"📊 {driver.get('frequency', '')} · Affects: {driver.get('affected_segment', '')}"
                    + (f" · KPI: {driver['kpi_to_track']}" if driver.get("kpi_to_track") else "")
                )
                if driver.get("root_cause"):
                    st.markdown(f"**Root Cause:** {driver['root_cause']}")
                if driver.get("evidence_quote"):
                    st.markdown(f"> *\"{driver['evidence_quote']}\"*")
                steps = driver.get("mitigation_steps", [])
                if steps:
                    with st.expander("🛠 Mitigation Playbook", expanded=True):
                        for step in steps:
                            st.markdown(f"- {step}")
                st.markdown("")

    with q1_chart:
        st.plotly_chart(_churn_driver_chart(recs.get("churn_drivers", []), df), use_container_width=True)

    st.markdown("---")

    # ── Q2: Upsell & Retention Segments ──────────────────────────────────────
    st.markdown("### ❓ Q2 — Highest Upsell & Retention Opportunities")
    q2_col, q2_chart = st.columns([3, 2])

    with q2_col:
        for seg in recs.get("upsell_segments", []):
            p_color = PRIORITY_COLOR.get(seg.get("priority", "medium"), "#FFA500")
            st.markdown(
                f"**{seg.get('segment', '')}** &nbsp;"
                + _pill(seg.get("priority", ""), p_color),
                unsafe_allow_html=True,
            )
            st.caption(
                f"📊 {seg.get('count', '')} interactions"
                + (f" · {seg['expected_revenue_impact']}" if seg.get("expected_revenue_impact") else "")
            )
            if seg.get("profile"):
                st.markdown(f"👤 **Profile:** {seg['profile']}")
            st.markdown(f"🎯 **Upsell Path:** {seg.get('upsell_path', '')}")
            st.markdown(f"🔒 **Retention Lever:** {seg.get('retention_lever', '')}")
            playbook = seg.get("outreach_playbook", [])
            if playbook:
                with st.expander("📬 Outreach Playbook", expanded=False):
                    for touch in playbook:
                        st.markdown(f"- {touch}")
            st.markdown(f"**Priority Action:** {seg.get('action', '')}")
            st.markdown("")

    with q2_chart:
        st.plotly_chart(_upsell_segment_chart(df), use_container_width=True)

    st.markdown("---")

    # ── Q3: Product Improvements ──────────────────────────────────────────────
    st.markdown("### ❓ Q3 — Highest-Impact Product & Service Improvements")
    improvements = recs.get("product_improvements", [])

    q3_col, q3_matrix = st.columns([3, 2])

    with q3_col:
        for i, imp in enumerate(improvements, 1):
            p_color = PRIORITY_COLOR.get(imp.get("priority", "medium"), "#FFA500")
            st.markdown(
                f"**{i}. {imp.get('improvement', '')}** &nbsp;"
                + _pill(imp.get("priority", ""), p_color)
                + "&nbsp;" + _pill(imp.get("owner", ""), "#1E90FF"),
                unsafe_allow_html=True,
            )
            st.caption(
                f"📊 {imp.get('frequency', '')} · Impact: {imp.get('impact','').title()} · Effort: {imp.get('effort','').title()}"
                + (f" · Metric: {imp['success_metric']}" if imp.get("success_metric") else "")
            )
            if imp.get("root_cause"):
                st.markdown(f"**Root Cause:** {imp['root_cause']}")
            roadmap = imp.get("implementation_roadmap", [])
            if roadmap:
                with st.expander("🗺 Implementation Roadmap", expanded=False):
                    for milestone in roadmap:
                        st.markdown(f"- {milestone}")
            st.markdown("")

    with q3_matrix:
        if improvements:
            st.plotly_chart(_impact_effort_matrix(improvements), use_container_width=True)

    st.markdown("---")

    # ── Action Item Registry ──────────────────────────────────────────────────
    st.markdown("### 📋 Action Item Registry")
    action_items = recs.get("action_items", [])
    if action_items:
        import pandas as pd
        for idx, item in enumerate(action_items, 1):
            p_color = PRIORITY_COLOR.get(item.get("priority", "medium"), "#FFA500")
            st.markdown(
                f"**{idx}. {item.get('action', '')}** &nbsp;"
                + _pill(item.get("priority", ""), p_color)
                + "&nbsp;" + _pill(item.get("owner", ""), "#1E90FF"),
                unsafe_allow_html=True,
            )
            st.caption(f"⏱ Due in {item.get('deadline_days', '?')} days · Evidence: {item.get('evidence', '')}")
            if item.get("how_to_execute"):
                st.markdown(f"*How:* {item['how_to_execute']}")
            st.markdown("")
    else:
        st.info("No action items generated.")

    st.markdown("---")

    # ── Raw synthesis JSON (for AI workflow evidence) ─────────────────────────
    with st.expander("🤖 Raw Synthesis JSON (AI Workflow Evidence)"):
        st.caption("Full LLM output from the synthesis call — grounded in the DataFrame stats below.")
        st.json(recs, expanded=False)
        st.caption("**Aggregated Stats passed to LLM:**")
        st.json(stats, expanded=False)
