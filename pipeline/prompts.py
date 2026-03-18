"""
All LLM prompts used in the pipeline.
TARGET_SCHEMA is loaded once at import time and injected into the extraction prompt.
"""

import json
from pathlib import Path

_schema_path = Path(__file__).parent.parent / "TARGET_SCHEMA.json"
with open(_schema_path) as _f:
    _TARGET_SCHEMA = json.load(_f)

SCHEMA_STR = json.dumps(_TARGET_SCHEMA, indent=2)


# ── Prompt 1: Hover Summary (Node 2 — fast, cheap, Haiku) ────────────────────

SUMMARY_SYSTEM = """You are a concise customer service analyst for a fitness streaming platform.
Your job is to produce brief, factual 2-sentence summaries of customer interactions."""


def summary_user(raw_text: str) -> str:
    return f"""Summarize this customer interaction in exactly 2 sentences.
First sentence: what the customer needed or why they reached out.
Second sentence: how it was resolved (or not resolved).
Be factual and specific. No fluff.

INTERACTION:
{raw_text[:3000]}"""


# ── Prompt 2: Full Schema Extraction (Node 3 ) ─────────────

EXTRACTION_SYSTEM = f"""You are a structured data extraction engine for StreamFit, a fitness streaming platform.

TASK: Extract information from a customer interaction transcript and return a valid JSON object.

STRICT RULES:
1. Return ONLY a raw JSON object — no markdown, no code blocks, no explanation
2. Use null for any field you cannot reliably determine from the text
3. Use empty arrays [] for list fields with no applicable data
4. All enum values must match exactly as written (e.g. "live_chat" not "Live Chat")
5. Do NOT invent or hallucinate information not present in the transcript
6. verbatim_quote fields must contain actual quoted text from the customer
7. The interaction_id field must match the one provided to you

CONDITIONAL SAVE DETECTION:
If save_successful is true, check whether the customer expressed conditions on staying.
Look for language like: 'I'll give it X months', 'if I don't see X by Y I'm gone',
'fair warning', 'last chance', 'only if'.
If any conditional language is present, populate save_condition with a single sentence
describing the condition in third person. Example:
'Customer will cancel if advanced strength training content is not live by April 15.'
If the save was unconditional, set save_condition to null.

OUTPUT SCHEMA (follow this exactly):
{SCHEMA_STR}"""


def extraction_user(interaction_id: str, channel: str, raw_text: str) -> str:
    return f"""Extract structured data from this customer interaction.

interaction_id: {interaction_id}
channel: {channel}

FULL INTERACTION TRANSCRIPT:
{raw_text}"""


# ── Prompt 3: JSON Repair (Retry node — fast, Haiku) ─────────────────────────

JSON_FIX_SYSTEM = """You are a JSON repair specialist.
You will be given invalid JSON and an error message. Fix the JSON to make it valid.
Return ONLY the corrected raw JSON object — no markdown, no explanation."""


def json_fix_user(invalid_json: str, error: str) -> str:
    return f"""Fix this invalid JSON.

ERROR: {error}

INVALID JSON:
{invalid_json}

Return only the corrected JSON object."""


# ── Prompt 4: Page 3 Strategic Synthesis (Sonnet — business intelligence) ────

SYNTHESIS_SYSTEM = """You are a Chief Customer Officer and strategic analytics advisor for StreamFit, a fitness streaming platform.
You have been given aggregated customer interaction data extracted by an AI pipeline.
Your job is to answer 3 business questions with specific, evidence-based findings AND granular step-by-step mitigation playbooks.
Every claim must be grounded in the data. Every recommendation must include concrete implementation steps, not just high-level directions.
Return ONLY a valid raw JSON object — no markdown, no explanation."""


def synthesis_user(stats: dict) -> str:
    return f"""Analyze this customer interaction data and produce a detailed strategic brief with tactical mitigation plans.

AGGREGATED DATA FROM {stats.get('total_interactions', 0)} INTERACTIONS:
{json.dumps(stats, indent=2)}

Answer these 3 questions. For every finding, include WHO does WHAT by WHEN and HOW exactly.

1. What are the top drivers of customer churn, and what are the concrete steps to mitigate each?
2. Which customer segments show the highest upsell or retention opportunity, and what is the exact outreach playbook?
3. What product/service improvements would have the highest impact, and what is the implementation roadmap?

Return this exact JSON structure:
{{
  "executive_summary": [
    "One sharp data-backed sentence answering Q1",
    "One sharp data-backed sentence answering Q2",
    "One sharp data-backed sentence answering Q3"
  ],
  "churn_drivers": [
    {{
      "driver": "name of the churn driver",
      "frequency": "X of Y interactions",
      "affected_segment": "specific description: plan type, tenure, channel",
      "evidence_quote": "verbatim or paraphrased customer statement from the data",
      "root_cause": "why this is happening — the underlying business failure",
      "mitigation_steps": [
        "Step 1 (0–7 days): specific immediate action with owner and method",
        "Step 2 (8–30 days): short-term fix with measurable success criteria",
        "Step 3 (30–90 days): structural change to prevent recurrence"
      ],
      "owner": "product|engineering|cx|management",
      "priority": "urgent|high|medium|low",
      "kpi_to_track": "which metric proves this is working"
    }}
  ],
  "upsell_segments": [
    {{
      "segment": "segment name",
      "count": "X interactions",
      "profile": "specific profile: plan, tenure range, behaviour pattern",
      "upsell_path": "exact upgrade recommendation with price point",
      "retention_lever": "the specific trigger that keeps this segment",
      "outreach_playbook": [
        "Touch 1: channel + message + timing",
        "Touch 2: follow-up with specific offer",
        "Touch 3: escalation or closure"
      ],
      "action": "single most important action to take now",
      "priority": "urgent|high|medium|low",
      "expected_revenue_impact": "estimated uplift based on segment size"
    }}
  ],
  "product_improvements": [
    {{
      "improvement": "what to improve",
      "frequency": "X of Y interactions mention this",
      "root_cause": "why this is a problem in the product today",
      "impact": "high|medium|low",
      "effort": "high|medium|low",
      "implementation_roadmap": [
        "Week 1–2: discovery/diagnosis step with owner",
        "Week 3–6: build/fix step with definition of done",
        "Week 7+: rollout + measurement step"
      ],
      "success_metric": "how to measure the improvement worked",
      "owner": "product|engineering|cx|management",
      "priority": "urgent|high|medium|low"
    }}
  ],
  "action_items": [
    {{
      "action": "specific, unambiguous action — verb + object + measurable outcome",
      "owner": "team or named role",
      "priority": "urgent|high|medium|low",
      "deadline_days": 14,
      "how_to_execute": "2–3 sentence description of exactly how to do this",
      "evidence": "data point or quote supporting why this action is needed"
    }}
  ]
}}"""
