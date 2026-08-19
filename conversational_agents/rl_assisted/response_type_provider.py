"""Deterministic response-type features used by deployment inference."""

RESPONSE_TYPE_LABELS = [
    "acknowledgment",
    "follow_up_question",
    "question",
    "statement",
    "confusion",
    "silence",
]


def predict_response_type(text: str) -> str:
    value = " ".join((text or "").lower().split())
    if not value:
        return "silence"
    if any(phrase in value for phrase in ("don't understand", "do not understand", "confused", "what do you mean")):
        return "confusion"
    if value.endswith("?"):
        return "question"
    if value in {"ok", "okay", "i see", "got it", "thanks", "thank you", "interesting"}:
        return "acknowledgment"
    return "statement"

