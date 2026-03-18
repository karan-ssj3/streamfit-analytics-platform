"""
Page 1 — Conversation Tiles

Grid of 6 interaction files. Each tile shows channel, ID, agent name and date.
Popover shows heuristic keyword signals — NO LLM call on page load or preview.
"Analyse All" button is the ONLY trigger for LLM extraction.
"""

import streamlit as st

from pipeline.loader import (
    list_all_interactions, extract_interaction_id,
    peek_agent_name, peek_signals, load_interaction,
)
from pipeline.graph import run_pipeline, load_cached, is_cached
from analysis.dataframe import build_dataframe, load_all_extracted

CHANNEL_ICON = {"phone": "📞", "live_chat": "💬", "email": "📧"}
RISK_CSS = {
    "none": "risk-none", "low": "risk-low", "medium": "risk-medium",
    "high": "risk-high", "immediate": "risk-immediate",
}


def _risk_class(interaction_id: str) -> str:
    """
    Return the CSS class name for the tile border colour based on cached churn risk level.

    Reads the churn risk level from the disk-cached extracted JSON (if available)
    and maps it to a CSS class defined in the app stylesheet.

    Args:
        interaction_id: Interaction ID string, e.g. 'SF-2026-0001'.

    Returns:
        CSS class string from RISK_CSS, e.g. 'risk-high'. Returns 'risk-unknown'
        if the interaction has not been analysed yet or the level is unrecognised.
    """
    cached = load_cached(interaction_id)
    if cached:
        level = (cached.get("intent", {}) or {}).get("churn_risk", {}).get("level", "unknown")
        return RISK_CSS.get(level, "risk-unknown")
    return "risk-unknown"


def _parse_header(raw_text: str) -> dict:
    """Extract metadata from the structured header block at the top of each .txt file."""
    meta = {"date": "", "duration": "", "channel": "", "agent": ""}
    for line in raw_text.splitlines()[:10]:
        for key in meta:
            if line.upper().startswith(key.upper() + ":"):
                meta[key] = line.split(":", 1)[1].strip()
    return meta


def render_page1():
    """
    Render the Page 1 tile grid — the main entry point of the StreamFit Audit Platform.

    Displays a 3-column grid of interaction tiles. Each tile shows the channel icon,
    interaction ID, agent name, date, duration, and analysis status badge. A popover
    on each tile shows heuristic keyword signals (no LLM call). If the interaction has
    already been analysed, the popover also shows churn risk, agent score, and sentiment.

    The 'Analyse All Files' button is the ONLY trigger for LLM extraction —
    no LLM calls happen during page load or popover display.
    """
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 🏃 StreamFit Audit Platform")
    st.markdown("**Customer Interaction Intelligence** — preview any conversation instantly. Click **Analyse All** to run the full AI extraction pipeline.")
    st.divider()

    all_files = list_all_interactions()
    total = len(all_files)
    cached_count = sum(1 for f in all_files if is_cached(extract_interaction_id(str(f))))

    # ── Top bar ───────────────────────────────────────────────────────────────
    col_stat1, col_stat2, col_stat3, col_cta = st.columns([1, 1, 1, 2])
    col_stat1.metric("Interactions", total)
    col_stat2.metric("Analysed", cached_count)
    col_stat3.metric("Pending", total - cached_count)

    with col_cta:
        st.write("")
        if st.button("⚡ Analyse All Files", type="primary", use_container_width=True):
            _run_batch_pipeline(all_files)

    st.divider()

    # ── Tile grid — 3 columns ─────────────────────────────────────────────────
    COLS = 3
    rows = [all_files[i:i + COLS] for i in range(0, len(all_files), COLS)]

    for row in rows:
        cols = st.columns(COLS)
        for col, fpath in zip(cols, row):
            interaction_id = extract_interaction_id(str(fpath))
            raw_text, channel = load_interaction(str(fpath))
            meta = _parse_header(raw_text)
            icon = CHANNEL_ICON.get(channel, "📄")
            risk_cls = _risk_class(interaction_id)
            cached = is_cached(interaction_id)
            badge = "✅" if cached else "⬜"
            agent_display = meta.get("agent") or peek_agent_name(raw_text)
            date_display = meta.get("date", "")
            duration_display = meta.get("duration", "")

            with col:
                st.markdown(f"""
                <div class="tile-card {risk_cls}">
                  <div style="font-size:1.4rem">{icon}</div>
                  <div style="font-weight:700; font-size:0.9rem; margin:4px 0">{interaction_id}</div>
                  <div style="font-size:0.75rem; color:#555">{channel.replace('_',' ').title()} · {date_display}</div>
                  <div style="font-size:0.75rem; color:#777; margin-top:4px">👤 {agent_display}</div>
                  <div style="font-size:0.72rem; color:#888">⏱ {duration_display}</div>
                  <div style="font-size:0.7rem; margin-top:6px">{badge} {'Analysed' if cached else 'Pending'}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Popover: keyword signals only — no LLM call ───────────────
                with st.popover("🔍 Preview", use_container_width=True):
                    st.markdown(f"**{interaction_id}** · {channel.replace('_', ' ').title()}")
                    st.caption(f"Agent: {agent_display} · {date_display} · {duration_display} min")
                    st.markdown("---")

                    # Keyword signals from heuristic scan (instant, no LLM)
                    signals = peek_signals(raw_text)
                    if signals:
                        signal_html = "&nbsp;&nbsp;".join(
                            f'<span style="background:#1e2a3a;color:#e0e0e0;padding:3px 9px;'
                            f'border-radius:12px;font-size:0.78rem">{icon} {label}</span>'
                            for icon, label in signals
                        )
                        st.markdown(signal_html, unsafe_allow_html=True)
                    else:
                        st.caption("No strong signals detected.")

                    st.markdown("")

                    # If already analysed, show churn risk badge from cache
                    cached_data = load_cached(interaction_id)
                    if cached_data:
                        churn_level = (cached_data.get("intent", {}) or {}).get("churn_risk", {}).get("level", "")
                        sentiment = (cached_data.get("sentiment", {}) or {}).get("overall", "")
                        resolution = (cached_data.get("interaction", {}) or {}).get("resolution", {}).get("status", "")
                        agent_score = cached_data.get("agent_quality_score", "")
                        cols_meta = st.columns(2)
                        if churn_level:
                            cols_meta[0].metric("Churn Risk", churn_level.title())
                        if agent_score:
                            cols_meta[1].metric("Agent Score", f"{agent_score}/10")
                        if sentiment:
                            st.caption(f"Sentiment: {sentiment.replace('_', ' ').title()} · Resolution: {resolution.replace('_', ' ').title()}")
                    else:
                        st.caption("_Run analysis to see extracted insights._")

                    st.markdown("")
                    if st.button(
                        "Open Full Analysis →",
                        key=f"open_{interaction_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.selected_id = interaction_id
                        st.session_state.page = "detail"
                        st.rerun()


def _run_batch_pipeline(all_files):
    """
    Run the full LangGraph extraction pipeline on all interaction files sequentially.

    Displays a Streamlit progress bar. Skips files that are already cached on disk.
    On completion, builds the flat DataFrame, stores it in session state, and navigates
    to the Page 2 dashboard.

    Args:
        all_files: List of Path objects pointing to the .txt transcript files to process.
    """
    progress = st.progress(0, text="Starting batch analysis…")
    total = len(all_files)

    for i, fpath in enumerate(all_files):
        interaction_id = extract_interaction_id(str(fpath))
        progress.progress(i / total, text=f"Analysing {interaction_id}… ({i + 1}/{total})")

        cached = load_cached(interaction_id)
        if not cached:
            run_pipeline(str(fpath))

    progress.progress(1.0, text="✅ All interactions analysed!")

    all_records = load_all_extracted()
    st.session_state.dataframe = build_dataframe(all_records)
    st.session_state.records_cache = all_records

    st.success(f"Analysed {total} interactions. Opening dashboard…")
    st.session_state.page = "dashboard"
    st.rerun()
