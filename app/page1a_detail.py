"""
Page 1A — Deep Analysis Spec Sheet

Left column : raw conversation text (scrollable, key quotes highlighted)
Right column : full spec sheet — 8 sections + collapsible JSON panel + raw LLM output tab
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from evaluation.evaluator import FieldAccuracyEvaluator
from pipeline.graph import run_pipeline, load_cached
from pipeline.loader import load_interaction, INTERACTIONS_DIR

REFERENCE_DIR = Path(__file__).parent.parent / "evaluation" / "reference"
EXTRACTED_DIR = Path(__file__).parent.parent / "extracted"

# ── Colour helpers ────────────────────────────────────────────────────────────
SENTIMENT_COLOR = {
    "very_positive": "#1a7a4a", "positive": "#2ECC71",
    "neutral": "#95A5A6", "mixed": "#FFA500",
    "negative": "#FF4B4B", "very_negative": "#8B0000",
}
CHURN_COLOR = {
    "none": "#2ECC71", "low": "#00CED1", "medium": "#FFA500",
    "high": "#FF4B4B", "immediate": "#8B0000",
}
UPSELL_COLOR = {"none": "#95A5A6", "low": "#00CED1", "medium": "#FFA500", "high": "#2ECC71"}
CONF_COLOR = {"high": "#2ECC71", "medium": "#FFA500", "low": "#FF4B4B"}
SEVERITY_COLOR = {"low": "#2ECC71", "medium": "#FFA500", "high": "#FF4B4B", "critical": "#8B0000"}


def _pill(text: str, color: str, text_color: str = "white") -> str:
    """
    Render an inline HTML badge/pill with a coloured background.

    Args:
        text:       Text to display inside the pill.
        color:      Background colour (hex or CSS colour string).
        text_color: Text colour. Defaults to 'white'.

    Returns:
        HTML string for a styled inline span element.
    """
    return (
        f'<span style="background:{color};color:{text_color};padding:3px 10px;'
        f'border-radius:20px;font-size:0.8rem;font-weight:600">{text}</span>'
    )


def _score_bar(score: float, max_score: float = 10.0) -> str:
    """
    Render an inline HTML progress bar for a numeric score.

    Colour thresholds: green (>= 7), amber (>= 5), red (< 5).

    Args:
        score:     The score value to display.
        max_score: The maximum possible score. Defaults to 10.0.

    Returns:
        HTML string for a styled div-based progress bar.
    """
    pct = min(score / max_score * 100, 100)
    color = "#2ECC71" if score >= 7 else "#FFA500" if score >= 5 else "#FF4B4B"
    return (
        f'<div style="background:#e0e0e0;border-radius:8px;height:10px;width:100%">'
        f'<div style="background:{color};width:{pct:.0f}%;height:10px;border-radius:8px"></div>'
        f'</div>'
    )


# ── Section renderers ─────────────────────────────────────────────────────────

def _section_snapshot(record: dict):
    """
    Render Section A — Interaction Snapshot with 4 headline metrics.

    Displays: interaction type, channel, duration, resolution status,
    resolution summary, agent name, handled_well flag, and notable actions.

    Args:
        record: Extracted interaction record dict (InteractionRecord.model_dump()).
    """
    st.markdown("#### 🔖 Interaction Snapshot")
    interaction = record.get("interaction", {})
    agent = interaction.get("agent", {})
    resolution = interaction.get("resolution", {})

    cols = st.columns(4)
    cols[0].metric("Type", interaction.get("type", "—").replace("_", " ").title())
    cols[1].metric("Channel", interaction.get("channel", "—").replace("_", " ").title())
    cols[2].metric("Duration", f"{interaction.get('duration_seconds', '—')}s" if interaction.get('duration_seconds') else "—")
    cols[3].metric("Resolution", resolution.get("status", "—").replace("_", " ").title())

    if resolution.get("summary"):
        st.caption(f"📋 {resolution['summary']}")
    st.markdown(f"**Agent:** {agent.get('name', '—')} &nbsp;{'✅ Handled well' if agent.get('handled_well') else '⚠️ Needs review'}", unsafe_allow_html=True)
    if agent.get("notable_actions"):
        st.caption(f"Notable: {agent['notable_actions']}")


def _section_problem(record: dict):
    """
    Render Section B — Core Problem Identification.

    Shows primary/secondary intent, the top pain point with severity badge and
    verbatim customer quote, and an expander for all additional pain points.

    Args:
        record: Extracted interaction record dict (InteractionRecord.model_dump()).
    """
    st.markdown("#### 🎯 Core Problem Identification")
    intent = record.get("intent", {})
    insights = record.get("insights", {})
    pain_points = insights.get("pain_points", [])

    st.markdown(f"**Primary Intent:** {intent.get('primary', '—')}")
    if intent.get("secondary"):
        st.markdown(f"**Secondary:** {intent['secondary']}")

    if pain_points:
        top = pain_points[0]
        sev_color = SEVERITY_COLOR.get(top.get("severity", "low"), "#95A5A6")
        st.markdown(
            f"**Top Pain Point:** {top.get('category', '—').replace('_', ' ').title()} &nbsp;"
            + _pill(top.get("severity", ""), sev_color),
            unsafe_allow_html=True,
        )
        st.markdown(f"> *\"{top.get('verbatim_quote', '')}\"*")
        st.caption(f"Actionable: {'✅ Yes' if top.get('actionable') else '❌ No'} — {top.get('description', '')}")

    if len(pain_points) > 1:
        with st.expander(f"View all {len(pain_points)} pain points"):
            for pp in pain_points:
                sev_color = SEVERITY_COLOR.get(pp.get("severity", "low"), "#95A5A6")
                st.markdown(
                    f"- **{pp.get('category','').replace('_',' ').title()}** "
                    + _pill(pp.get("severity",""), sev_color)
                    + f" — {pp.get('description','')}"
                    + (f"\n  > *\"{pp.get('verbatim_quote','')}\"*" if pp.get("verbatim_quote") else ""),
                    unsafe_allow_html=True,
                )


def _section_customer(record: dict):
    """
    Render Section C — Customer Profile in a 2-row, 3-column metric grid.

    Displays: name, plan, tenure, lifecycle stage, fitness level, age range,
    and household (if available).

    Args:
        record: Extracted interaction record dict (InteractionRecord.model_dump()).
    """
    st.markdown("#### 👤 Customer Profile")
    customer = record.get("customer", {})
    demo = customer.get("demographic_signals", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Name", customer.get("name") or "Unknown")
    c2.metric("Plan", (customer.get("current_plan") or "—").replace("_", " ").title())
    c3.metric("Tenure", f"{customer.get('tenure_months', '?')} months")

    c4, c5, c6 = st.columns(3)
    c4.metric("Lifecycle", (customer.get("lifecycle_stage") or "—").replace("_", " ").title())
    c5.metric("Fitness Level", (demo.get("fitness_level") or "—").title())
    c6.metric("Age Range", demo.get("age_range") or "—")

    if demo.get("household"):
        st.caption(f"🏠 Household: {demo['household']}")


def _section_sentiment(record: dict):
    """
    Render Section D — Sentiment Analysis with key moments timeline.

    Shows: overall sentiment pill (colour-coded), trajectory arrow, emotional
    intensity label, and a chronological list of key moments with shift icons,
    timestamps, descriptions, and triggers.

    Args:
        record: Extracted interaction record dict (InteractionRecord.model_dump()).
    """
    st.markdown("#### 💬 Sentiment Analysis")
    sentiment = record.get("sentiment", {})
    overall = sentiment.get("overall", "neutral")
    trajectory = sentiment.get("trajectory", "stable")
    intensity = sentiment.get("emotional_intensity", "low")
    key_moments = sentiment.get("key_moments", [])

    traj_arrow = {"improving": "↑ Improving", "stable": "→ Stable", "declining": "↓ Declining"}.get(trajectory, trajectory)
    color = SENTIMENT_COLOR.get(overall, "#95A5A6")

    st.markdown(
        _pill(overall.replace("_", " ").title(), color)
        + f" &nbsp; {traj_arrow} &nbsp; Intensity: **{intensity.title()}**",
        unsafe_allow_html=True,
    )

    if key_moments:
        st.markdown("**Key Moments:**")
        for km in key_moments:
            shift_icon = {"positive_spike": "🟢", "negative_spike": "🔴", "turning_point": "🔄"}.get(km.get("sentiment_shift", ""), "⚪")
            ts = f"`{km['timestamp']}`" if km.get("timestamp") else ""
            st.markdown(f"{shift_icon} {ts} **{km.get('description', '')}** — *trigger: {km.get('trigger', '')}*")


def _section_churn_upsell(record: dict):
    """
    Render Section E — Churn Risk & Upsell Signal in a 2-column layout.

    Left column: churn risk level badge, contributing factors, save attempt
    outcome, and a conditional save warning (st.warning) if save_condition is set.
    Right column: upsell opportunity badge, target plan, and upsell signals.

    Args:
        record: Extracted interaction record dict (InteractionRecord.model_dump()).
    """
    st.markdown("#### ⚡ Churn Risk & Upsell Signal")
    intent = record.get("intent", {})
    churn = intent.get("churn_risk", {})
    upsell = intent.get("upsell_opportunity", {})

    c1, c2 = st.columns(2)

    with c1:
        churn_level = churn.get("level", "none")
        churn_color = CHURN_COLOR.get(churn_level, "#95A5A6")
        st.markdown(f"**Churn Risk:** " + _pill(churn_level.upper(), churn_color), unsafe_allow_html=True)
        factors = churn.get("factors", [])
        for f in factors:
            st.markdown(f"  • {f}")
        if churn.get("save_attempted"):
            saved = churn.get("save_successful")
            st.markdown(f"  💼 Save attempted → {'✅ Successful' if saved else '❌ Unsuccessful' if saved is False else '⏳ Pending'}")
            if churn.get("save_condition"):
                st.warning(f"⚠️ Conditional Save — {churn['save_condition']}")

    with c2:
        upsell_level = upsell.get("level", "none")
        upsell_color = UPSELL_COLOR.get(upsell_level, "#95A5A6")
        st.markdown(f"**Upsell Opportunity:** " + _pill(upsell_level.upper(), upsell_color), unsafe_allow_html=True)
        if upsell.get("target_plan"):
            st.markdown(f"  🎯 Target: **{upsell['target_plan']}**")
        for sig in upsell.get("signals", []):
            st.markdown(f"  • {sig}")


def _section_agent_scorecard(record: dict):
    """
    Render Section F — Agent Evaluation Scorecard in a 2-column layout.

    Left column: agent_quality_score with progress bar and sub-dimension breakdown
    (empathy, resolution status, notable actions).
    Right column: conversation_quality_score with progress bar and data quality metrics
    (interaction quality, extraction confidence).

    Args:
        record: Extracted interaction record dict (InteractionRecord.model_dump()).
    """
    st.markdown("#### 🎖 Agent Evaluation Scorecard")
    agent_score = record.get("agent_quality_score", 0.0)
    conv_score = record.get("conversation_quality_score", 0.0)
    interaction = record.get("interaction", {})
    agent = interaction.get("agent", {})
    quality = record.get("quality_flags", {})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Agent Quality Score: {agent_score}/10**")
        st.markdown(_score_bar(agent_score), unsafe_allow_html=True)
        st.caption("Based on: handled_well · resolution status · action completion · data quality")

        # Sub-dimension breakdown
        handled = agent.get("handled_well", False)
        res_status = interaction.get("resolution", {}).get("status", "unresolved")
        st.markdown(f"- Empathy: {'✅' if handled else '⚠️'} {'Good' if handled else 'Needs improvement'}")
        st.markdown(f"- Resolution: **{res_status.replace('_',' ').title()}**")
        if agent.get("notable_actions"):
            st.markdown(f"- Notable: _{agent['notable_actions']}_")

    with c2:
        st.markdown(f"**Conversation Quality Score: {conv_score}/10**")
        st.markdown(_score_bar(conv_score), unsafe_allow_html=True)
        st.caption("Based on: extraction confidence · field completeness · sentiment trajectory")

        iq = quality.get("interaction_quality", "—")
        ec = quality.get("extraction_confidence", "—")
        st.markdown(f"- Data quality: **{iq.replace('_',' ').title()}**")
        st.markdown(f"- Extraction confidence: " + _pill(ec, CONF_COLOR.get(ec, "#95A5A6")), unsafe_allow_html=True)


def _section_action_items(record: dict):
    """
    Render Section G — Action Items list with owner and priority badges.

    Each action is displayed with a status icon (✅/⏳/🔄), owner pill,
    and priority pill (colour-coded urgent → low). Does nothing if no action items exist.

    Args:
        record: Extracted interaction record dict (InteractionRecord.model_dump()).
    """
    action_items = record.get("action_items", [])
    if not action_items:
        return
    st.markdown("#### ✅ Action Items")
    PRIORITY_COLOR = {"urgent": "#8B0000", "high": "#FF4B4B", "medium": "#FFA500", "low": "#2ECC71"}
    STATUS_ICON = {"completed": "✅", "pending": "⏳", "requires_follow_up": "🔄"}

    for ai in action_items:
        p_color = PRIORITY_COLOR.get(ai.get("priority", "low"), "#95A5A6")
        s_icon = STATUS_ICON.get(ai.get("status", "pending"), "⏳")
        st.markdown(
            f"{s_icon} **{ai.get('action', '')}** &nbsp;"
            + _pill(ai.get("owner", ""), "#1E90FF")
            + " &nbsp;"
            + _pill(ai.get("priority", ""), p_color),
            unsafe_allow_html=True,
        )


def _section_quality_flags(record: dict):
    """
    Render Section H — Quality Flags in a 3-column metric layout.

    Displays: extraction confidence pill, interaction quality label, and
    human review required (red Yes / green No). Lists any review reason captions below.

    Args:
        record: Extracted interaction record dict (InteractionRecord.model_dump()).
    """
    st.markdown("#### 🚦 Quality Flags")
    quality = record.get("quality_flags", {})
    ec = quality.get("extraction_confidence", "low")
    iq = quality.get("interaction_quality", "significant_gaps")
    review = quality.get("requires_human_review", False)

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Extraction Confidence**<br>" + _pill(ec, CONF_COLOR.get(ec, "#95A5A6")), unsafe_allow_html=True)
    c2.markdown(f"**Interaction Quality**<br>{iq.replace('_',' ').title()}", unsafe_allow_html=True)
    c3.markdown(f"**Human Review Required**<br>{'🔴 Yes' if review else '🟢 No'}", unsafe_allow_html=True)

    reasons = quality.get("review_reasons", [])
    if reasons:
        for r in reasons:
            st.caption(f"⚠️ {r}")


def _section_other_insights(record: dict):
    """
    Render Section I — Additional Insights as collapsible expanders.

    Shows motivations, feature requests, competitor mentions, and topics, each
    in their own st.expander — only rendered if the respective list is non-empty.
    Competitor sentiment and feature urgency are colour-coded with pills.

    Args:
        record: Extracted interaction record dict (InteractionRecord.model_dump()).
    """
    insights = record.get("insights", {})
    competitors = insights.get("competitor_mentions", [])
    motivations = insights.get("motivations", [])
    features = insights.get("feature_requests", [])
    topics = record.get("topics", [])

    if motivations:
        with st.expander(f"💪 Motivations ({len(motivations)})"):
            for m in motivations:
                strength_icon = {"strong": "🔥", "moderate": "✅", "weak": "💤"}.get(m.get("strength",""), "")
                st.markdown(f"{strength_icon} **{m.get('type','').replace('_',' ').title()}** — {m.get('description','')}")

    if features:
        with st.expander(f"🔧 Feature Requests ({len(features)})"):
            for fr in features:
                urg_color = {"dealbreaker": "#FF4B4B", "important": "#FFA500", "nice_to_have": "#2ECC71"}.get(fr.get("urgency",""), "#95A5A6")
                st.markdown(
                    f"- **{fr.get('feature','')}** &nbsp;" + _pill(fr.get("urgency","").replace("_"," "), urg_color),
                    unsafe_allow_html=True,
                )
                if fr.get("existing_workaround"):
                    st.caption(f"  Workaround: {fr['existing_workaround']}")

    if competitors:
        with st.expander(f"⚔️ Competitor Mentions ({len(competitors)})"):
            for cm in competitors:
                sent_color = {"competitor_preferred": "#FF4B4B", "neutral": "#FFA500", "we_preferred": "#2ECC71"}.get(cm.get("sentiment_vs_us",""), "#95A5A6")
                st.markdown(
                    f"- **{cm.get('name','')}** ({cm.get('context','').replace('_',' ')}) &nbsp;"
                    + _pill(cm.get("sentiment_vs_us","").replace("_"," "), sent_color)
                    + f" — {cm.get('detail','')}",
                    unsafe_allow_html=True,
                )

    if topics:
        with st.expander(f"🏷 Topics ({len(topics)})"):
            for t in topics:
                sent_dot = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(t.get("sentiment",""), "⚪")
                rel = _pill(t.get("relevance",""), "#1E90FF")
                st.markdown(f"{sent_dot} {t.get('topic','')} &nbsp;{rel}", unsafe_allow_html=True)


# ── Main render ───────────────────────────────────────────────────────────────

def render_page1a():
    """
    Render the Page 1A deep analysis spec sheet for a single interaction.

    Expects st.session_state['selected_id'] to be set (set by clicking a tile on Page 1).
    Loads the record from session state or disk cache; runs the full pipeline if not cached.

    Layout:
        Left column (40%): raw conversation text in a scrollable monospace panel.
        Right column (60%): tabbed view with:
            - Tab 1 (Spec Sheet): 9 spec sections A–I
            - Tab 2 (Populated Schema): full JSON + Schema Extensions expander
            - Tab 3 (Raw LLM Output): pre-Pydantic model output for auditability
            - Tab 4 (Extraction Eval): accuracy evaluation vs reference (if reference exists)
    """
    interaction_id = st.session_state.get("selected_id")
    if not interaction_id:
        st.error("No interaction selected.")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    # Navigation bar
    nav_l, nav_r = st.columns([1, 8])
    with nav_l:
        if st.button("← Back"):
            st.session_state.page = "home"
            st.rerun()
    with nav_r:
        st.markdown(f"### 📋 {interaction_id} — Deep Analysis")

    # Load or run pipeline
    record = st.session_state.extracted.get(interaction_id) or load_cached(interaction_id)
    if record is None:
        file_path = str(INTERACTIONS_DIR / f"{interaction_id}.txt")
        with st.spinner(f"Running extraction pipeline on {interaction_id}…"):
            record = run_pipeline(file_path)
        if record:
            st.session_state.extracted[interaction_id] = record

    if not record:
        st.error("Extraction failed for this interaction. Please try again.")
        return

    raw_text, _ = load_interaction(str(INTERACTIONS_DIR / f"{interaction_id}.txt"))

    # ── Two-column layout ────────────────────────────────────────────────────
    col_raw, col_spec = st.columns([4, 6])

    with col_raw:
        st.markdown("#### 📄 Raw Conversation")
        st.markdown(
            f'<div style="height:75vh;overflow-y:auto;background:#f5f5f5;padding:16px;'
            f'border-radius:8px;font-size:0.82rem;font-family:monospace;white-space:pre-wrap;'
            f'border:1px solid #ddd">{raw_text}</div>',
            unsafe_allow_html=True,
        )

    with col_spec:
        # Build tab list — add Extraction Eval tab if a reference file exists for this interaction
        has_eval = (REFERENCE_DIR / f"{interaction_id}_reference.json").exists()
        tab_labels = ["📊 Spec Sheet", "🗂 Populated Schema", "🤖 Raw LLM Output"]
        if has_eval:
            tab_labels.append("📊 Extraction Eval")
        tabs = st.tabs(tab_labels)
        tab_spec, tab_json, tab_raw = tabs[0], tabs[1], tabs[2]
        tab_eval = tabs[3] if has_eval else None

        with tab_spec:
            _section_snapshot(record)
            st.divider()
            _section_problem(record)
            st.divider()
            _section_customer(record)
            st.divider()
            _section_sentiment(record)
            st.divider()
            _section_churn_upsell(record)
            st.divider()
            _section_agent_scorecard(record)
            st.divider()
            _section_action_items(record)
            st.divider()
            _section_quality_flags(record)
            st.divider()
            _section_other_insights(record)

        with tab_json:
            st.caption("Full populated TARGET_SCHEMA — exactly what the production pipeline writes to storage.")
            # Show clean version without raw_llm_output
            display = {k: v for k, v in record.items() if k != "raw_llm_output"}
            st.json(display, expanded=False)

            with st.expander("ℹ️ Schema Extensions"):
                st.caption("These four fields are intentional additions beyond TARGET_SCHEMA.json — not schema drift.")
                import pandas as pd
                ext_df = pd.DataFrame([
                    {"Field": "agent_quality_score",        "Type": "float 0–10", "Source": "Node 5 deterministic",    "Reason": "Enables agent benchmarking across interactions"},
                    {"Field": "conversation_quality_score", "Type": "float 0–10", "Source": "Node 5 deterministic",    "Reason": "Measures data richness independently of agent quality"},
                    {"Field": "hover_summary",              "Type": "string",     "Source": "Node 2 LLM (gpt-4o-mini)", "Reason": "Instant tile preview without re-reading transcript"},
                    {"Field": "raw_llm_output",             "Type": "string",     "Source": "Node 3 raw output",       "Reason": "Full auditability of extraction step pre-Pydantic"},
                ])
                st.dataframe(ext_df, use_container_width=True, hide_index=True)

        with tab_raw:
            st.caption("Raw LLM output string before Pydantic parsing. Shows what the model actually returned.")
            raw_output = record.get("raw_llm_output", "")
            if raw_output:
                st.code(raw_output, language="json")
            else:
                st.info("Raw output not available (loaded from cache without it).")

        if tab_eval is not None:
            with tab_eval:
                ref_path = str(REFERENCE_DIR / f"{interaction_id}_reference.json")
                ext_path = str(EXTRACTED_DIR / f"{interaction_id}.json")

                if not Path(ext_path).exists():
                    st.warning("Run the extraction pipeline first — no cached JSON found for this interaction.")
                else:
                    ev = FieldAccuracyEvaluator(ref_path, ext_path)
                    results = ev.evaluate()
                    breakdown_df = ev.to_dataframe()

                    # ── Overall accuracy ──────────────────────────────────────
                    pct = round(results["overall_accuracy"] * 100, 1)
                    acc_col, m_col, p_col, x_col = st.columns(4)
                    acc_col.metric("Overall Accuracy", f"{pct}%")
                    m_col.metric("✅ Exact Matches", results["exact_matches"])
                    p_col.metric("🟡 Partial", results["partial_matches"])
                    x_col.metric("❌ Mismatches", results["mismatches"])

                    st.divider()

                    # ── Section score bar chart ───────────────────────────────
                    section_scores = results["section_scores"]
                    fig = go.Figure(go.Bar(
                        x=list(section_scores.values()),
                        y=[s.replace("_", " ").title() for s in section_scores.keys()],
                        orientation="h",
                        marker_color=[
                            "#2ECC71" if v >= 0.85 else "#FFA500" if v >= 0.60 else "#FF4B4B"
                            for v in section_scores.values()
                        ],
                        text=[f"{v*100:.0f}%" for v in section_scores.values()],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Section-Level Accuracy",
                        xaxis=dict(range=[0, 1.15], tickformat=".0%"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", size=12),
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=260,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

                    # ── Field breakdown table ─────────────────────────────────
                    st.markdown("#### Field Breakdown")
                    st.dataframe(
                        breakdown_df[["Section", "Field", "Expected", "Extracted", "Score", "Match", "Notes"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        "Evaluated against manually annotated reference for SF-2026-0001. "
                        "In production, reference annotations would be provided by the QA team "
                        "for sampled interactions, creating a continuous evaluation loop."
                    )

    # ── Footer nav ────────────────────────────────────────────────────────────
    st.divider()
    if st.button("→ Open Dashboard (Page 2)", type="primary"):
        st.session_state.page = "dashboard"
        st.rerun()
