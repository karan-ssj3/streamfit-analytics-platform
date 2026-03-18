"""
Page 2 — Consolidated Analytics Dashboard

Loads all extracted JSON files, builds a flat DataFrame, renders 12 Plotly charts
across 4 sections. Button at bottom opens Page 3 (LLM Recommender).
"""

import streamlit as st
import pandas as pd

from analysis.dataframe import (
    build_dataframe,
    load_all_extracted,
    get_feature_requests_table,
    get_competitor_table,
)
from analysis.charts import (
    churn_risk_bar,
    lifecycle_stage_donut,
    sentiment_by_channel,
    top_pain_categories,
    pain_severity_heatmap,
    feature_request_chart,
    agent_quality_bar,
    resolution_status_donut,
    save_attempts_funnel,
    upsell_scatter,
    competitor_tracker,
    high_value_at_risk_table,
)


def _load_data() -> tuple[pd.DataFrame, list[dict]]:
    """
    Load or build the interactions DataFrame and raw records list.

    Checks session state first to avoid re-reading disk. If not cached,
    loads all extracted JSON files from disk, builds the flat DataFrame,
    and stores both in session state for subsequent page visits.

    Returns:
        Tuple of (df, records) where df is the flat Pandas DataFrame and
        records is the list of raw extracted dicts.
    """
    if st.session_state.get("dataframe") is not None and not st.session_state["dataframe"].empty:
        df = st.session_state["dataframe"]
        records = st.session_state.get("records_cache") or load_all_extracted()
    else:
        records = load_all_extracted()
        df = build_dataframe(records)
        st.session_state["dataframe"] = df
        st.session_state["records_cache"] = records
    return df, records


def render_page2():
    """
    Render the Page 2 consolidated analytics dashboard across 4 sections.

    Requires extracted interactions to be present on disk. Shows a warning and
    exits early if no data is found.

    Sections:
        1. Customer Health Overview: churn risk bar, lifecycle donut, sentiment by channel.
        2. Pain Points & Issues: pain categories bar, severity heatmap, feature request chart.
        3. Agent & Resolution Performance: agent quality bar, resolution donut, save funnel.
        4. Opportunity Signals: upsell scatter, competitor tracker, high-value at risk table.

    Also includes: 6 headline KPI metrics, agent scorecard expander, raw DataFrame expander,
    and a navigation button to Page 3.
    """
    # ── Header ────────────────────────────────────────────────────────────────
    nav_l, nav_r = st.columns([1, 8])
    with nav_l:
        if st.button("← Home"):
            st.session_state.page = "home"
            st.rerun()
    with nav_r:
        st.markdown("### 📊 Consolidated Interaction Dashboard")

    df, records = _load_data()

    if df.empty:
        st.warning("No extracted interactions found yet. Go to Page 1 and click **Analyse All Files**.")
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Interactions", len(df))
    k2.metric("High Churn Risk", int((df["churn_risk_score"] >= 3).sum()),
              delta=f"{(df['churn_risk_score'] >= 3).mean()*100:.0f}%", delta_color="inverse")
    k3.metric("Upsell Opportunities", int((df["upsell_score"] >= 2).sum()),
              delta=f"{(df['upsell_score'] >= 2).mean()*100:.0f}%")
    k4.metric("Avg Agent Score", f"{df['agent_quality_score'].mean():.1f}/10")
    k5.metric("Resolved", f"{(df['resolution_status'] == 'resolved').mean()*100:.0f}%")
    k6.metric("Needs Review", int(df["requires_human_review"].sum()),
              delta_color="inverse")

    st.divider()

    # ── Section 1: Customer Health ────────────────────────────────────────────
    st.markdown("### 1️⃣ Customer Health Overview")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.plotly_chart(churn_risk_bar(df), use_container_width=True)
    with r1c2:
        st.plotly_chart(lifecycle_stage_donut(df), use_container_width=True)
    with r1c3:
        st.plotly_chart(sentiment_by_channel(df), use_container_width=True)

    st.divider()

    # ── Section 2: Pain Points & Issues ──────────────────────────────────────
    st.markdown("### 2️⃣ Pain Points & Issues")
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.plotly_chart(top_pain_categories(df), use_container_width=True)
    with r2c2:
        st.plotly_chart(pain_severity_heatmap(df), use_container_width=True)
    with r2c3:
        feat_df = get_feature_requests_table(records)
        st.plotly_chart(feature_request_chart(feat_df), use_container_width=True)

    # Feature request detail table
    if not feat_df.empty:
        with st.expander("📋 Full Feature Request List"):
            st.dataframe(
                feat_df.rename(columns={"feature": "Feature", "count": "Mentions", "avg_urgency": "Avg Urgency"}),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # ── Section 3: Agent & Resolution Performance ─────────────────────────────
    st.markdown("### 3️⃣ Agent & Resolution Performance")
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        st.plotly_chart(agent_quality_bar(df), use_container_width=True)
    with r3c2:
        st.plotly_chart(resolution_status_donut(df), use_container_width=True)
    with r3c3:
        st.plotly_chart(save_attempts_funnel(df), use_container_width=True)

    # Agent scorecard table
    with st.expander("📋 Agent Scorecard Detail"):
        scorecard = (
            df.groupby("agent_name")
            .agg(
                Interactions=("interaction_id", "count"),
                Avg_Agent_Score=("agent_quality_score", "mean"),
                Avg_Conv_Score=("conv_quality_score", "mean"),
                Resolved=("resolution_status", lambda x: f"{(x=='resolved').mean()*100:.0f}%"),
                Save_Rate=("save_successful", lambda x: f"{x.mean()*100:.0f}%" if x.notna().any() else "N/A"),
            )
            .reset_index()
            .rename(columns={
                "agent_name": "Agent",
                "Avg_Agent_Score": "Avg Agent Score",
                "Avg_Conv_Score": "Avg Conv Score",
                "Resolved": "Resolution Rate",
                "Save_Rate": "Save Rate",
            })
        )
        scorecard["Avg Agent Score"] = scorecard["Avg Agent Score"].round(2)
        scorecard["Avg Conv Score"] = scorecard["Avg Conv Score"].round(2)
        st.dataframe(scorecard, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 4: Opportunity Signals ────────────────────────────────────────
    st.markdown("### 4️⃣ Opportunity Signals")
    r4c1, r4c2 = st.columns([3, 2])
    with r4c1:
        st.plotly_chart(upsell_scatter(df), use_container_width=True)
    with r4c2:
        comp_df = get_competitor_table(records)
        st.plotly_chart(competitor_tracker(comp_df), use_container_width=True)

    # High-value at-risk table
    at_risk_df = high_value_at_risk_table(df)
    if not at_risk_df.empty:
        st.markdown("#### 🚨 High-Value Customers At Risk")
        st.dataframe(
            at_risk_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("✅ No high-value customers (premium/family, tenure > 12 months) currently at high churn risk.")

    st.divider()

    # ── Raw DataFrame ─────────────────────────────────────────────────────────
    with st.expander("🗃 Raw Interactions DataFrame"):
        display_cols = [
            "interaction_id", "channel", "type", "agent_name",
            "customer_plan", "lifecycle_stage", "churn_risk_level",
            "upsell_level", "sentiment_overall", "agent_quality_score",
            "extraction_confidence",
        ]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
        st.caption(f"{len(df)} rows × {len(df.columns)} columns total")

    # ── Footer nav ─────────────────────────────────────────────────────────────
    st.divider()
    if st.button("→ Get Strategic Recommendations (Page 3)", type="primary"):
        st.session_state.page = "recommender"
        st.rerun()
