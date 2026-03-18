"""
evaluation/evaluator.py

Field-level accuracy scoring: compares pipeline-extracted JSON against a
manually annotated reference file.

Usage:
    ev = FieldAccuracyEvaluator(reference_path, extracted_path)
    results = ev.evaluate()
    df = ev.to_dataframe()
"""

import json
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ── Field registry ─────────────────────────────────────────────────────────────
# Each entry: (section, display_name, dot_path, score_type)
# dot_path is the same in both reference and extracted (Pydantic model_dump keys).
# score_type: "exact" | "numeric" | "array" | "presence"

FIELDS = [
    # Interaction
    ("interaction", "type",               "interaction.type",                        "exact"),
    ("interaction", "channel",            "interaction.channel",                     "exact"),
    ("interaction", "resolution.status",  "interaction.resolution.status",           "exact"),
    ("interaction", "agent.handled_well", "interaction.agent.handled_well",          "exact"),
    # Customer
    ("customer",    "current_plan",       "customer.current_plan",                   "exact"),
    ("customer",    "lifecycle_stage",    "customer.lifecycle_stage",                "exact"),
    ("customer",    "tenure_months",      "customer.tenure_months",                  "numeric"),
    ("customer",    "fitness_level",      "customer.demographic_signals.fitness_level", "exact"),
    # Sentiment
    ("sentiment",   "overall",            "sentiment.overall",                       "exact"),
    ("sentiment",   "trajectory",         "sentiment.trajectory",                    "exact"),
    ("sentiment",   "emotional_intensity","sentiment.emotional_intensity",            "exact"),
    # Intent
    ("intent",      "churn_risk.level",       "intent.churn_risk.level",             "exact"),
    ("intent",      "save_attempted",         "intent.churn_risk.save_attempted",    "exact"),
    ("intent",      "save_successful",        "intent.churn_risk.save_successful",   "exact"),
    ("intent",      "save_condition",         "intent.churn_risk.save_condition",    "presence"),
    ("intent",      "upsell.level",           "intent.upsell_opportunity.level",     "exact"),
    # Quality flags
    ("quality_flags", "extraction_confidence",   "quality_flags.extraction_confidence",  "exact"),
    ("quality_flags", "requires_human_review",   "quality_flags.requires_human_review",  "exact"),
]


def _get(obj: dict, dot_path: str) -> Any:
    """Traverse a nested dict using a dot-notation path. Returns None on missing keys."""
    parts = dot_path.split(".")
    cur = obj
    for p in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _normalise(val: Any) -> Any:
    """Normalise values for comparison: lowercase strings, keep other types as-is."""
    if isinstance(val, str):
        return val.lower().strip()
    return val


# ── Scoring functions ──────────────────────────────────────────────────────────

def _score_exact(expected: Any, actual: Any) -> tuple[float, str]:
    """1.0 on exact match (case-insensitive strings), 0.0 otherwise. Null-aware."""
    if expected is None and actual is None:
        return 1.0, "both null — correct abstention"
    if expected is None or actual is None:
        return 0.0, f"null mismatch (expected={expected}, actual={actual})"
    if _normalise(expected) == _normalise(actual):
        return 1.0, "exact match"
    return 0.0, f"mismatch"


def _score_numeric(expected: Any, actual: Any) -> tuple[float, str]:
    """1.0 within 10%, 0.5 within 25%, 0.0 otherwise. Null-aware."""
    if expected is None and actual is None:
        return 1.0, "both null — correct abstention"
    if expected is None or actual is None:
        return 0.0, f"null mismatch (expected={expected}, actual={actual})"
    try:
        e, a = float(expected), float(actual)
    except (TypeError, ValueError):
        return 0.0, "non-numeric value"
    if e == 0:
        return (1.0, "exact match") if a == 0 else (0.0, f"expected 0, got {a}")
    pct_diff = abs(e - a) / abs(e)
    if pct_diff <= 0.10:
        return 1.0, f"within 10% (expected={e}, actual={a}, diff={pct_diff:.1%})"
    if pct_diff <= 0.25:
        return 0.5, f"within 25% (expected={e}, actual={a}, diff={pct_diff:.1%})"
    return 0.0, f"outside 25% tolerance (expected={e}, actual={a}, diff={pct_diff:.1%})"


def _score_array(expected: Any, actual: Any) -> tuple[float, str]:
    """Jaccard similarity on string-normalised elements. Null-aware."""
    if expected is None and actual is None:
        return 1.0, "both null — correct abstention"
    if expected is None or actual is None:
        return 0.0, f"null mismatch"
    if not isinstance(expected, list):
        expected = [expected]
    if not isinstance(actual, list):
        actual = [actual]
    e_set = {_normalise(str(x)) for x in expected}
    a_set = {_normalise(str(x)) for x in actual}
    if not e_set and not a_set:
        return 1.0, "both empty"
    intersection = len(e_set & a_set)
    union = len(e_set | a_set)
    jaccard = intersection / union if union > 0 else 0.0
    return round(jaccard, 3), f"Jaccard={jaccard:.2f} (|∩|={intersection}, |∪|={union})"


def _score_presence(expected: Any, actual: Any) -> tuple[float, str]:
    """
    For fields where exact string match is not meaningful (e.g. free-text save_condition).
    Checks: both non-null → 1.0, both null → 1.0, mismatch in nullness → 0.0.
    """
    e_present = expected is not None and expected != ""
    a_present = actual is not None and actual != ""
    if e_present == a_present:
        return 1.0, "presence matches" if e_present else "both absent — correct abstention"
    return 0.0, f"presence mismatch (expected present={e_present}, actual present={a_present})"


SCORERS = {
    "exact": _score_exact,
    "numeric": _score_numeric,
    "array": _score_array,
    "presence": _score_presence,
}


# ── Evaluator class ────────────────────────────────────────────────────────────

class FieldAccuracyEvaluator:
    """
    Compares a pipeline-extracted JSON against a human-annotated reference JSON.
    Computes per-field and per-section accuracy scores.

    Scoring rules:
    - Exact match (strings, enums, booleans): 1.0 or 0.0
    - Numeric match within 10%: 1.0, within 25%: 0.5, else 0.0
    - Null handling: both null → 1.0 (correct abstention), one null → 0.0
    - Arrays: Jaccard similarity on string elements
    - Presence: both non-null or both null → 1.0 (for free-text fields)

    Sections evaluated:
    - interaction  (type, channel, resolution.status, agent.handled_well)
    - customer     (current_plan, lifecycle_stage, tenure_months, fitness_level)
    - sentiment    (overall, trajectory, emotional_intensity)
    - intent       (churn_risk.level, save_attempted, save_successful, save_condition, upsell.level)
    - quality_flags (extraction_confidence, requires_human_review)
    """

    def __init__(self, reference_path: str, extracted_path: str):
        """
        Initialise the evaluator by loading both JSON files.

        Args:
            reference_path: Path to the human-annotated ground truth JSON file.
            extracted_path: Path to the pipeline-extracted JSON file to evaluate.
        """
        with open(reference_path) as f:
            self.reference = json.load(f)
        with open(extracted_path) as f:
            self.extracted = json.load(f)
        self._results: Optional[dict] = None

    def evaluate(self) -> dict:
        """
        Run full field-level evaluation across all 18 registered fields.

        Compares each field in the extracted JSON against the reference JSON using the
        appropriate scorer (exact/numeric/array/presence). Aggregates results into
        per-section scores and an overall accuracy score.

        Returns:
            Dict with keys:
                - interaction_id (str)
                - overall_accuracy (float 0.0–1.0)
                - section_scores (dict[section_name, float])
                - field_breakdown (list of per-field result dicts)
                - total_fields (int)
                - exact_matches (int): fields with score == 1.0
                - partial_matches (int): fields with 0 < score < 1.0
                - mismatches (int): fields with score == 0.0
        """
        field_breakdown = []

        for section, display_name, dot_path, score_type in FIELDS:
            expected = _get(self.reference, dot_path)
            actual = _get(self.extracted, dot_path)
            score_fn = SCORERS[score_type]
            score, notes = score_fn(expected, actual)

            field_breakdown.append({
                "section": section,
                "field": display_name,
                "expected": expected,
                "extracted": actual,
                "score": score,
                "match": "✅" if score == 1.0 else ("🟡" if score > 0 else "❌"),
                "notes": notes,
                "score_type": score_type,
            })

        # Section scores: mean of fields within each section
        section_scores: dict[str, float] = {}
        section_order = []
        for section, _, _, _ in FIELDS:
            if section not in section_scores:
                section_scores[section] = 0.0
                section_order.append(section)

        for section in section_order:
            section_fields = [r for r in field_breakdown if r["section"] == section]
            section_scores[section] = round(
                sum(r["score"] for r in section_fields) / len(section_fields), 3
            )

        overall_accuracy = round(
            sum(r["score"] for r in field_breakdown) / len(field_breakdown), 3
        )

        self._results = {
            "interaction_id": self.reference.get("interaction_id", "unknown"),
            "overall_accuracy": overall_accuracy,
            "section_scores": section_scores,
            "field_breakdown": field_breakdown,
            "total_fields": len(field_breakdown),
            "exact_matches": sum(1 for r in field_breakdown if r["score"] == 1.0),
            "partial_matches": sum(1 for r in field_breakdown if 0 < r["score"] < 1.0),
            "mismatches": sum(1 for r in field_breakdown if r["score"] == 0.0),
        }
        return self._results

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the field breakdown results as a display-ready Pandas DataFrame.

        Calls evaluate() if it hasn't been run yet. Renames internal keys to
        human-readable column names (Section, Field, Expected, Extracted, Score, Match, Notes).

        Returns:
            DataFrame with one row per evaluated field and 7 columns.
        """
        if self._results is None:
            self.evaluate()
        rows = []
        for r in self._results["field_breakdown"]:
            rows.append({
                "Section": r["section"].replace("_", " ").title(),
                "Field": r["field"],
                "Expected": str(r["expected"]) if r["expected"] is not None else "null",
                "Extracted": str(r["extracted"]) if r["extracted"] is not None else "null",
                "Score": r["score"],
                "Match": r["match"],
                "Notes": r["notes"],
            })
        return pd.DataFrame(rows)
