from __future__ import annotations

import re
from dataclasses import dataclass


RQ4_YES_NO = {"yes", "no"}
RQ4_MCQ = {"a", "b", "c", "d"}
RQ4_HINDI_MCQ = {"ए": "a", "बी": "b", "सी": "c", "डी": "d"}
RQ4_HINDI_YES_NO = {"हाँ": "yes", "हां": "yes", "नहीं": "no", "नही": "no"}


@dataclass(frozen=True)
class RQ4Evaluation:
    predicted: str | None
    status: str  # correct | incorrect | unparseable
    score: float


def _clean_output(model_output: str) -> str:
    cleaned = str(model_output or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:text|txt)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def _explicit_marker_answer(cleaned: str, gold_clean: str) -> str | None:
    if gold_clean in RQ4_MCQ:
        marker = re.findall(
            r"(?:answer|उत्तर|final answer)\s*[:\-]?\s*\(?([ABCD])\)?\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if marker:
            return marker[-1].lower()

        hindi_marker = re.findall(
            r"(?:उत्तर|जवाब)\s*[:\-]?\s*(ए|बी|सी|डी)",
            cleaned,
        )
        if hindi_marker:
            return RQ4_HINDI_MCQ[hindi_marker[-1]]

    if gold_clean in RQ4_YES_NO:
        marker = re.findall(
            r"(?:answer|उत्तर|final answer)\s*[:\-]?\s*(yes|no|हाँ|हां|नहीं|नही)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if marker:
            token = marker[-1].lower()
            return RQ4_HINDI_YES_NO.get(token, token)

    return None


def _leading_option_answer(cleaned: str) -> str | None:
    """Extract an MCQ label when it occurs at the start of the response.

    This accepts natural model outputs such as `A. Entailment` or
    `C) Contradiction`, while deliberately refusing arbitrary A/B/C/D letters
    that occur later inside a reasoning trace.
    """
    match = re.match(r"^\s*\(?([ABCD])\s*[.)\-:]\s*", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()

    hindi_match = re.match(r"^\s*\(?([एबीसीडी])\s*[.)\-:]\s*", cleaned)
    if hindi_match:
        return RQ4_HINDI_MCQ[hindi_match.group(1)]

    return None


def extract_rq4_answer(model_output: str, gold: str) -> str | None:
    """Strict RQ4 extraction.

    Accepted MCQ forms include bare labels (`B`), leading option labels with
    option text (`B. Entailment`), and explicit answer markers (`Answer: B`).
    We never infer an answer from arbitrary letters appearing later in a
    verbose reasoning trace.
    """
    cleaned = _clean_output(model_output)
    gold_clean = str(gold).strip().lower()

    if not cleaned:
        return None

    explicit = _explicit_marker_answer(cleaned, gold_clean)
    if explicit is not None:
        return explicit

    if gold_clean in RQ4_MCQ:
        leading = _leading_option_answer(cleaned)
        if leading is not None:
            return leading

        bare = re.fullmatch(r"\(?\s*([ABCD])\s*[.)]?\s*", cleaned, flags=re.IGNORECASE)
        if bare:
            return bare.group(1).lower()

        bare_hindi = re.fullmatch(r"\(?\s*(ए|बी|सी|डी)\s*[.)]?\s*", cleaned)
        if bare_hindi:
            return RQ4_HINDI_MCQ[bare_hindi.group(1)]

        return None

    if gold_clean in RQ4_YES_NO:
        bare = re.fullmatch(r"\s*(yes|no)\s*[.!]?\s*", cleaned, flags=re.IGNORECASE)
        if bare:
            return bare.group(1).lower()

        bare_hindi = re.fullmatch(r"\s*(हाँ|हां|नहीं|नही)\s*[.!]?\s*", cleaned)
        if bare_hindi:
            return RQ4_HINDI_YES_NO[bare_hindi.group(1)]

        return None

    if re.fullmatch(r"\d+", gold_clean):
        bare_number = re.fullmatch(r"\s*(\d+)\s*[.]?\s*", cleaned)
        return bare_number.group(1) if bare_number else None

    return None


def evaluate_rq4(model_output: str, gold: str) -> RQ4Evaluation:
    predicted = extract_rq4_answer(model_output, gold)
    gold_clean = str(gold).strip().lower()

    if predicted is None:
        return RQ4Evaluation(None, "unparseable", 0.0)

    if predicted == gold_clean:
        return RQ4Evaluation(predicted, "correct", 1.0)

    return RQ4Evaluation(predicted, "incorrect", 0.0)
