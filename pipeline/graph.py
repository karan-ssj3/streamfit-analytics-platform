"""
LangGraph extraction pipeline.
6 nodes: load → summarize → extract → validate (with retry) → score → persist

Uses OpenAI GPT-4o (extraction) and GPT-4o-mini (summaries / JSON fixes).
LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true is set in .env.

In production this same graph runs inside Azure Container Apps workers
with Azure OpenAI replacing the direct OpenAI endpoint.
"""

import json
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import OpenAI
from pydantic import ValidationError

from .loader import load_interaction, extract_interaction_id
from .models import InteractionRecord
from .models import (
    ActionOwner, ActionPriority, ActionStatus,
    Channel, ChurnRisk, ChurnRiskLevel, Customer,
    DemographicSignals, ExtractionConfidence, Insights, Intent,
    InteractionMeta, InteractionQuality, InteractionRecord,
    InteractionType, Agent, Resolution, ResolutionStatus,
    PipelineState, QualityFlags, Sentiment, SentimentOverall,
    SentimentTrajectory, EmotionalIntensity, UpsellLevel, UpsellOpportunity,
    LifecycleStage,
)
from .prompts import (
    EXTRACTION_SYSTEM, extraction_user,
    JSON_FIX_SYSTEM, json_fix_user,
    SUMMARY_SYSTEM, summary_user,
)

load_dotenv()

EXTRACTED_DIR = Path(__file__).parent.parent / "extracted"
EXTRACTED_DIR.mkdir(exist_ok=True)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """
    Return the singleton OpenAI client, initialising it on first call.

    Reads OPENAI_API_KEY from environment (loaded via python-dotenv).

    Returns:
        Shared OpenAI client instance.
    """
    global _client
    if _client is None:
        _client = OpenAI()   # reads OPENAI_API_KEY from env
    return _client


def _chat(system: str, user: str, model: str = "gpt-4o-mini", max_tokens: int = 300) -> str:
    """
    Thin wrapper around OpenAI chat completions — single system + user turn.

    Args:
        system:     System prompt string (role instructions for the model).
        user:       User message string (the actual content/question).
        model:      OpenAI model ID. Defaults to 'gpt-4o-mini' for cost efficiency.
        max_tokens: Maximum tokens in the response. Defaults to 300.

    Returns:
        Stripped response string from the model's first choice.
    """
    resp = get_client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def _strip_fences(text: str) -> str:
    """
    Remove markdown code fences (```json ... ```) from an LLM response string.

    Args:
        text: Raw LLM output that may be wrapped in triple-backtick code fences.

    Returns:
        Clean string with fences stripped and leading/trailing whitespace removed.
    """
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    return text


# ── Node 1: Loader ────────────────────────────────────────────────────────────

def node_load(state: PipelineState) -> dict:
    """
    Node 1 — Load raw transcript file and detect channel type.

    Reads the .txt file at state['file_path'], detects the channel (phone/live_chat/email),
    and extracts the interaction ID from the filename.

    Args:
        state: PipelineState dict. Requires 'file_path' to be set.

    Returns:
        Partial state update with keys: raw_text, channel, interaction_id,
        retry_count (reset to 0), errors (reset to []).
    """
    raw_text, channel = load_interaction(state["file_path"])
    interaction_id = extract_interaction_id(state["file_path"])
    return {
        "raw_text": raw_text,
        "channel": channel,
        "interaction_id": interaction_id,
        "retry_count": 0,
        "errors": [],
    }


# ── Node 2: Summarizer ────────────────────────────────────────────────────────

def node_summarize(state: PipelineState) -> dict:
    """2-sentence hover card summary. GPT-4o-mini for cost efficiency."""
    try:
        summary = _chat(
            system=SUMMARY_SYSTEM,
            user=summary_user(state["raw_text"]),
            model="gpt-4o-mini",
            max_tokens=200,
        )
    except Exception:
        summary = "Summary unavailable."
    return {"summary": summary}


# ── Node 3: Extractor ─────────────────────────────────────────────────────────

def node_extract(state: PipelineState) -> dict:
    """Full schema extraction. GPT-4o for accuracy."""
    try:
        raw_output = _chat(
            system=EXTRACTION_SYSTEM,
            user=extraction_user(state["interaction_id"], state["channel"], state["raw_text"]),
            model="gpt-4o",
            max_tokens=4096,
        )
        raw_output = _strip_fences(raw_output)
    except Exception as e:
        return {
            "raw_llm_output": "{}",
            "errors": state.get("errors", []) + [f"LLM extraction error: {e}"],
        }
    return {"raw_llm_output": raw_output}


# ── Node 4: Validator ─────────────────────────────────────────────────────────

def node_validate(state: PipelineState) -> dict:
    """
    Node 4 — Parse and validate the LLM JSON output using Pydantic.

    Parses raw_llm_output as JSON, falls back to using the pipeline interaction_id
    if it's missing from the extracted dict, then validates against InteractionRecord.
    On success, also runs detect_conditional_save() as a post-validation hook.
    On failure, increments retry_count (triggers the fix → retry loop up to 2 times).

    Args:
        state: PipelineState dict. Requires 'raw_llm_output' and 'interaction_id'.

    Returns:
        On success: state update with extracted_dict and validated_record (model_dump).
        On failure: state update with empty extracted_dict, None validated_record,
                    incremented retry_count, and appended errors.
    """
    raw = state.get("raw_llm_output", "{}")
    try:
        data = json.loads(raw)
        if not data.get("interaction_id"):
            data["interaction_id"] = state["interaction_id"]
        record = InteractionRecord.model_validate(data)
        # Deterministic post-validation check — catches conditional saves the LLM may miss
        record = detect_conditional_save(state.get("raw_text", ""), record)
        return {
            "extracted_dict": record.model_dump(),
            "validated_record": record.model_dump(),
            "errors": state.get("errors", []),
        }
    except (json.JSONDecodeError, ValidationError) as e:
        errors = state.get("errors", []) + [str(e)]
        return {
            "extracted_dict": {},
            "validated_record": None,
            "errors": errors,
            "retry_count": state.get("retry_count", 0) + 1,
        }


# ── Node: JSON Fixer (retry path) ─────────────────────────────────────────────

def node_fix_json(state: PipelineState) -> dict:
    """
    Retry node — ask GPT-4o-mini to repair a broken JSON string.

    Called when node_validate fails. Sends the malformed raw_llm_output and the
    last error message to the model with instructions to return valid JSON only.
    The corrected output is written back to raw_llm_output so node_validate can retry.

    Args:
        state: PipelineState dict. Requires 'raw_llm_output' and 'errors' to be set.

    Returns:
        State update with corrected 'raw_llm_output', or appended error on failure.
    """
    error_msg = state["errors"][-1] if state.get("errors") else "Invalid JSON"
    try:
        fixed = _chat(
            system=JSON_FIX_SYSTEM,
            user=json_fix_user(state.get("raw_llm_output", "{}"), error_msg),
            model="gpt-4o-mini",
            max_tokens=4096,
        )
        return {"raw_llm_output": _strip_fences(fixed)}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"JSON fix error: {e}"]}


# ── Node: Fallback (max retries exceeded) ────────────────────────────────────

def node_fallback(state: PipelineState) -> dict:
    """
    Fallback node — construct a minimal valid InteractionRecord when all retries are exhausted.

    Called after 2 failed validation attempts. Creates a stub record with
    extraction_confidence=low and requires_human_review=True, preserving the
    interaction_id and channel so the record can still be persisted and reviewed.

    Args:
        state: PipelineState dict. Uses 'interaction_id', 'channel', and 'errors'.

    Returns:
        State update with a valid (but minimal) validated_record and extracted_dict.
    """
    record = InteractionRecord(
        interaction_id=state.get("interaction_id", "unknown"),
        interaction=InteractionMeta(
            type=InteractionType.GENERAL_INQUIRY,
            channel=Channel(state.get("channel", "phone")),
            agent=Agent(name="Unknown", handled_well=False),
            resolution=Resolution(
                status=ResolutionStatus.UNRESOLVED,
                summary="Automated extraction failed — requires manual review.",
            ),
        ),
        customer=Customer(lifecycle_stage=LifecycleStage.ACTIVE),
        sentiment=Sentiment(
            overall=SentimentOverall.NEUTRAL,
            trajectory=SentimentTrajectory.STABLE,
            emotional_intensity=EmotionalIntensity.LOW,
        ),
        insights=Insights(),
        intent=Intent(
            primary="Unknown — extraction failed",
            churn_risk=ChurnRisk(level=ChurnRiskLevel.NONE),
            upsell_opportunity=UpsellOpportunity(level=UpsellLevel.NONE),
        ),
        quality_flags=QualityFlags(
            interaction_quality=InteractionQuality.SIGNIFICANT_GAPS,
            extraction_confidence=ExtractionConfidence.LOW,
            requires_human_review=True,
            review_reasons=[
                "Automated extraction failed after retries: "
                + "; ".join(state.get("errors", []))[:200]
            ],
        ),
    )
    return {
        "validated_record": record.model_dump(),
        "extracted_dict": record.model_dump(),
    }


# ── Node 5: Scorer ────────────────────────────────────────────────────────────

# Weight rationale:
# Resolution (3.5)          — the single most important signal. Did the agent actually solve the problem?
# Interaction Quality (2.5) — clean, complete transcripts reflect agent professionalism.
# Empathy (2.0)             — important for CX but a secondary quality signal.
# Action Follow-through (2.0) — rewards agents who close items, but outcome > process.
# Total ceiling: 10.0. partially_resolved can score max ~8.0, not 9.0.

def node_score(state: PipelineState) -> dict:
    """
    Node 5 — Compute agent_quality_score and conversation_quality_score (both 0–10).

    Agent Quality (10 pts total):
        - Resolution status (3.5 pts): resolved=3.5, partially_resolved=1.8, escalated=1.2, cancelled=0.5, unresolved=0.0
        - Interaction quality (2.5 pts): clean=2.5, minor_issues=1.5, significant_gaps=0.5
        - Empathy/handled_well (2.0 pts): handled_well=2.0, else=0.8
        - Action follow-through (2.0 pts): (completed_actions / total_actions) * 2.0

    Conversation Quality (10 pts total):
        - Extraction confidence (4.0 pts): high=4.0, medium=2.5, low=1.0
        - Field completeness (3.0 pts): proportion of customer/sentiment/insights/intent filled
        - Sentiment trajectory (3.0 pts): improving=3.0, stable=2.0, declining=1.0

    Args:
        state: PipelineState dict. Requires 'validated_record' (model_dump dict).

    Returns:
        State update with 'scores' dict containing agent_quality_score and conversation_quality_score.
    """
    r = state.get("validated_record")
    if not r:
        return {"scores": {"agent_quality_score": 0.0, "conversation_quality_score": 0.0}}

    interaction = r.get("interaction", {})
    quality_flags = r.get("quality_flags", {})
    action_items = r.get("action_items", [])

    # ── Agent Quality ──────────────────────────────────────────────────────────

    # Resolution (3.5 pts) — primary outcome; highest weight because it's what customers measure.
    resolution_status = interaction.get("resolution", {}).get("status", "unresolved")
    resolution_map = {
        "resolved": 3.5,            # full credit — problem solved
        "partially_resolved": 1.8,  # < half ceiling; still left work on the table
        "escalated": 1.2,           # agent passed it up — partial credit for triage
        "cancelled": 0.5,           # interaction aborted; minimal value
        "unresolved": 0.0,          # worst outcome; no credit
    }
    res_score = resolution_map.get(resolution_status, 0.0)

    # Interaction Quality (2.5 pts) — unchanged; clean transcripts reflect agent discipline.
    iq = quality_flags.get("interaction_quality", "significant_gaps")
    iq_map = {"clean": 2.5, "minor_issues": 1.5, "significant_gaps": 0.5}
    iq_score = iq_map.get(iq, 0.5)

    # Empathy (2.0 pts) — reduced from 2.5; CX quality matters but outcome matters more.
    handled_well = interaction.get("agent", {}).get("handled_well", False)
    empathy = 2.0 if handled_well else 0.8  # partial credit even when not handled well

    # Action Follow-through (2.0 pts) — reduced from 2.5; rewards closure but ranks below outcome.
    completed = sum(1 for a in action_items if a.get("status") == "completed")
    total = len(action_items) if action_items else 1
    action_score = (completed / total) * 2.0

    agent_quality = min(round(res_score + iq_score + empathy + action_score, 1), 10.0)

    # Conversation Quality
    confidence = quality_flags.get("extraction_confidence", "low")
    conf_map = {"high": 4.0, "medium": 2.5, "low": 1.0}
    key_sections = ["customer", "sentiment", "insights", "intent"]
    filled = sum(1 for s in key_sections if r.get(s))
    completeness = (filled / len(key_sections)) * 3.0
    trajectory = r.get("sentiment", {}).get("trajectory", "stable")
    traj_map = {"improving": 3.0, "stable": 2.0, "declining": 1.0}
    conv_quality = min(round(conf_map.get(confidence, 1.0) + completeness + traj_map.get(trajectory, 2.0), 1), 10.0)

    return {"scores": {
        "agent_quality_score": agent_quality,
        "conversation_quality_score": conv_quality,
    }}


# ── Conditional save scanner ──────────────────────────────────────────────────

CONDITIONAL_SAVE_PATTERNS = [
    r"if (i|we) don.t see",
    r"give it \w+ months",
    r"fair warning",
    r"last chance",
    r"gone for good",
    r"only if",
    r"unless",
    r"by (january|february|march|april|may|june|july|august|september|october|november|december)",
]


def detect_conditional_save(raw_text: str, record: InteractionRecord) -> InteractionRecord:
    """
    Post-extraction deterministic check.
    If save_successful=True but raw text contains conditional language,
    ensure requires_human_review=True and add a specific review reason.
    This catches cases where the LLM correctly marked save_successful=True
    but missed the conditional nature of the retention.
    """
    if record.intent.churn_risk.save_successful:
        for pattern in CONDITIONAL_SAVE_PATTERNS:
            if re.search(pattern, raw_text, re.IGNORECASE):
                review_reason = (
                    "Conditional save detected: customer's retention is time-bound or conditional. "
                    "Follow-up required before save_condition deadline."
                )
                if review_reason not in record.quality_flags.review_reasons:
                    record.quality_flags.review_reasons.append(review_reason)
                record.quality_flags.requires_human_review = True
                break
    return record


# ── Node 6: Persister ─────────────────────────────────────────────────────────

def node_persist(state: PipelineState) -> dict:
    """
    Node 6 — Merge scores and metadata into the record and write to disk as JSON.

    Adds agent_quality_score, conversation_quality_score, hover_summary, and
    raw_llm_output to the validated record dict, then writes it to
    /extracted/{interaction_id}.json. In production, this node writes to Azure Data Lake.

    Args:
        state: PipelineState dict. Requires 'validated_record', 'scores', 'summary',
               'raw_llm_output', and 'interaction_id'.

    Returns:
        State update with the final merged 'validated_record'.
    """
    record = dict(state.get("validated_record", {}))
    scores = state.get("scores", {})
    record["agent_quality_score"] = scores.get("agent_quality_score", 0.0)
    record["conversation_quality_score"] = scores.get("conversation_quality_score", 0.0)
    record["hover_summary"] = state.get("summary", "")
    record["raw_llm_output"] = state.get("raw_llm_output", "")

    output_path = EXTRACTED_DIR / f"{state['interaction_id']}.json"
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return {"validated_record": record}


# ── Conditional routing ───────────────────────────────────────────────────────

def route_after_validate(state: PipelineState) -> str:
    """
    Conditional edge function — decides the next node after validation.

    Routes to 'score' on success, 'fix' if retry budget remains, or 'fallback'
    after 2 failed retries.

    Args:
        state: PipelineState dict. Checks 'validated_record' and 'retry_count'.

    Returns:
        Next node name: 'score' | 'fix' | 'fallback'.
    """
    if state.get("validated_record") is not None:
        return "score"
    if state.get("retry_count", 0) < 2:
        return "fix"
    return "fallback"


# ── Build + compile the graph ─────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the 6-node LangGraph extraction DAG.

    Graph topology:
        load → summarize → extract → validate → score → persist
                                         ↑↓ fix (retry, max 2)
                                         ↓ fallback (after 2 failures)

    Returns:
        Compiled LangGraph runnable (CompiledGraph).
    """
    g = StateGraph(PipelineState)
    g.add_node("load",     node_load)
    g.add_node("summarize", node_summarize)
    g.add_node("extract",  node_extract)
    g.add_node("validate", node_validate)
    g.add_node("fix",      node_fix_json)
    g.add_node("fallback", node_fallback)
    g.add_node("score",    node_score)
    g.add_node("persist",  node_persist)

    g.set_entry_point("load")
    g.add_edge("load",      "summarize")
    g.add_edge("summarize", "extract")
    g.add_edge("extract",   "validate")
    g.add_conditional_edges(
        "validate", route_after_validate,
        {"score": "score", "fix": "fix", "fallback": "fallback"},
    )
    g.add_edge("fix",      "validate")
    g.add_edge("fallback", "score")
    g.add_edge("score",    "persist")
    g.add_edge("persist",  END)
    return g.compile()


_graph = None


def get_graph():
    """
    Return the singleton compiled LangGraph, building it on first call.

    Returns:
        Compiled LangGraph runnable (CompiledGraph).
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(file_path: str) -> dict:
    """
    Run the full 6-node extraction pipeline for a single interaction file.

    Initialises a fresh PipelineState, invokes the compiled LangGraph, and
    returns the final validated + scored record. Output is also written to
    /extracted/{interaction_id}.json by node_persist.

    Args:
        file_path: Path to the .txt transcript file to process.

    Returns:
        Final validated_record dict (InteractionRecord.model_dump() + extension fields),
        or empty dict if the pipeline fails catastrophically.
    """
    initial: PipelineState = {
        "file_path": file_path,
        "interaction_id": "",
        "raw_text": "",
        "channel": "",
        "summary": "",
        "raw_llm_output": "",
        "extracted_dict": {},
        "validated_record": None,
        "scores": {},
        "retry_count": 0,
        "errors": [],
    }
    result = get_graph().invoke(initial)
    return result.get("validated_record", {})


def get_summary_only(file_path: str) -> str:
    """
    Fast path — generate a 2-sentence hover card summary using GPT-4o-mini only.

    Skips all other pipeline nodes. Used when only a quick summary is needed
    without running full schema extraction.

    Args:
        file_path: Path to the .txt transcript file.

    Returns:
        2-sentence summary string, or 'Summary unavailable.' on error.
    """
    raw_text, _ = load_interaction(file_path)
    try:
        return _chat(
            system=SUMMARY_SYSTEM,
            user=summary_user(raw_text),
            model="gpt-4o-mini",
            max_tokens=200,
        )
    except Exception:
        return "Summary unavailable."


def load_cached(interaction_id: str) -> Optional[dict]:
    """
    Load a previously extracted record from disk cache.

    Args:
        interaction_id: Interaction ID string, e.g. 'SF-2026-0001'.

    Returns:
        Parsed JSON dict from /extracted/{interaction_id}.json, or None if not cached.
    """
    path = EXTRACTED_DIR / f"{interaction_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def is_cached(interaction_id: str) -> bool:
    """
    Check whether a given interaction has already been extracted and saved to disk.

    Args:
        interaction_id: Interaction ID string, e.g. 'SF-2026-0001'.

    Returns:
        True if /extracted/{interaction_id}.json exists, False otherwise.
    """
    return (EXTRACTED_DIR / f"{interaction_id}.json").exists()
