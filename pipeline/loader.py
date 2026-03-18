"""
File loader and channel detector.
Reads raw .txt interaction files and infers the channel type (phone/live_chat/email).
"""

import re
from pathlib import Path
from typing import Tuple

INTERACTIONS_DIR = Path(__file__).parent.parent / "interactions"


def load_interaction(file_path: str) -> Tuple[str, str]:
    """
    Load raw text from a .txt interaction file and detect its channel type.

    Args:
        file_path: Absolute or relative path to the .txt transcript file.

    Returns:
        Tuple of (raw_text, channel) where channel is 'phone', 'live_chat', or 'email'.
    """
    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8")
    channel = detect_channel(raw_text)
    return raw_text, channel


def detect_channel(raw_text: str) -> str:
    """
    Detect interaction channel from transcript content using keyword heuristics.

    Checks for email signals (Subject:/From: headers), phone signals (HH:MM:SS
    timestamps, 'call transcript'), and live chat signals (wall-clock timestamps
    with agent names). Defaults to 'phone' if no signal is found.

    Args:
        raw_text: Full raw transcript text as a string.

    Returns:
        Channel string: 'phone' | 'live_chat' | 'email'.
    """
    text_lower = raw_text.lower()

    # Email signals
    email_signals = ["subject:", "from:", "dear ", "regards,", "sincerely,", "hi there,", "email thread", "cc:", "bcc:"]
    if sum(1 for s in email_signals if s in text_lower) >= 2:
        return "email"

    # Phone call signals — elapsed time format [HH:MM:SS] or "call transcript"
    if re.search(r"\[\d{2}:\d{2}:\d{2}\]", raw_text):
        return "phone"
    phone_signals = ["phone call", "call transcript", "call recording", "hold", "transfer", "dial"]
    if any(s in text_lower for s in phone_signals):
        return "phone"

    # Live chat signals — wall-clock timestamps or chat-specific markers
    if re.search(r"\d{2}:\d{2}:\d{2}.*(?:user|agent|customer|support|james|sarah|marcus)", text_lower):
        return "live_chat"
    chat_signals = ["live chat", "chat session", "chat transcript", "user:", "typing"]
    if any(s in text_lower for s in chat_signals):
        return "live_chat"

    # Default to phone (most common in this dataset)
    return "phone"


ACTIVE_INTERACTIONS = 6  # demo uses first 6 files only


def list_all_interactions() -> list[Path]:
    """
    Return a sorted list of the first ACTIVE_INTERACTIONS interaction .txt files.

    Returns:
        List of Path objects pointing to SF-*.txt files, capped at ACTIVE_INTERACTIONS (6).
    """
    return sorted(INTERACTIONS_DIR.glob("SF-*.txt"))[:ACTIVE_INTERACTIONS]


def peek_signals(raw_text: str) -> list[tuple[str, str]]:
    """
    Heuristic keyword scan of raw transcript — returns up to 5 (icon, label) signal tuples.
    No LLM call. Pure text matching against 9 known domain keyword categories.

    Categories checked (in order): Churn Risk, Competitor Mention, Billing,
    Content Gap, Tech Issue, High Emotion, Upsell Signal, Save Offer, Positive.

    Args:
        raw_text: Full raw transcript text as a string.

    Returns:
        List of up to 5 tuples of (emoji_icon, label_string), e.g. [('🚨', 'Churn Risk')].
    """
    text_lower = raw_text.lower()
    signals: list[tuple[str, str]] = []

    if any(w in text_lower for w in ["cancel", "cancellation", "leaving", "quit", "unsubscribe", "close my account"]):
        signals.append(("🚨", "Churn Risk"))
    if any(w in text_lower for w in ["fitflow", "competitor", "switching to", "trying out", "moved to", "comparing"]):
        signals.append(("⚠️", "Competitor Mention"))
    if any(w in text_lower for w in ["billing", "charge", "refund", "invoice", "overcharged", "payment", "price increase"]):
        signals.append(("💳", "Billing"))
    if any(w in text_lower for w in ["stale", "same workout", "no new content", "content hasn", "library hasn", "nothing new"]):
        signals.append(("📚", "Content Gap"))
    if any(w in text_lower for w in ["app crash", "doesn't work", "error", "bug", "freeze", "samsung", "playback", "loading", "buffering"]):
        signals.append(("🔧", "Tech Issue"))
    if any(w in text_lower for w in ["frustrated", "upset", "angry", "ridiculous", "unacceptable", "terrible", "awful", "furious"]):
        signals.append(("😤", "High Emotion"))
    if any(w in text_lower for w in ["upgrade", "annual plan", "family plan", "add another user", "premium"]):
        signals.append(("🎯", "Upsell Signal"))
    if any(w in text_lower for w in ["discount", "offer", "half price", "loyalty", "50%", "three months"]):
        signals.append(("💡", "Save Offer"))
    if any(w in text_lower for w in ["thank you", "appreciate", "very helpful", "great service", "happy with", "love the"]):
        signals.append(("✅", "Positive"))

    return signals[:5]


def extract_interaction_id(file_path: str) -> str:
    """
    Extract the interaction ID from a file path by returning the stem (filename without extension).

    Args:
        file_path: Path to the transcript file, e.g. '/interactions/SF-2026-0001.txt'.

    Returns:
        Interaction ID string, e.g. 'SF-2026-0001'.
    """
    return Path(file_path).stem


def get_interaction_date(file_path: str) -> str:
    """
    Extract a human-readable display date from the filename (best-effort).

    Parses the year and sequence number from the SF-YYYY-NNNN filename pattern.
    Does not read file content.

    Args:
        file_path: Path to the transcript file, e.g. '/interactions/SF-2026-0001.txt'.

    Returns:
        Display string like '2026 · #0001', or empty string if pattern doesn't match.
    """
    stem = Path(file_path).stem  # SF-2026-0001
    parts = stem.split("-")
    if len(parts) >= 2:
        return f"2026 · #{parts[-1]}"
    return ""


def peek_agent_name(raw_text: str) -> str:
    """
    Quick heuristic to extract the agent name from the first 30 lines of a transcript
    without running a full LLM call.

    Looks for lines containing 'AGENT:', 'REPRESENTATIVE:', or 'SUPPORT AGENT:' headers,
    or spoken phrases like 'My name is X', 'This is X', or 'I'm X'.

    Args:
        raw_text: Full raw transcript text as a string.

    Returns:
        Agent name string if found (e.g. 'Sarah Mitchell'), or 'Agent' as fallback.
    """
    for line in raw_text.splitlines()[:30]:
        line_lower = line.lower()
        if "agent:" in line_lower or "representative:" in line_lower or "support agent:" in line_lower:
            # Extract the name after the colon
            parts = line.split(":", 1)
            if len(parts) == 2:
                name = parts[1].strip().split("\n")[0].strip()
                if name and len(name) < 40:
                    return name
        # Look for "My name is X" or "This is X"
        match = re.search(r"(?:my name is|this is|i'm|speaking with)\s+([A-Z][a-z]+ [A-Z][a-z]+)", line)
        if match:
            return match.group(1)
    return "Agent"
