from __future__ import annotations

import re
from dataclasses import dataclass


RQ4_YES_NO = {"yes", "no"}
RQ4_MCQ = {"a", "b", "c", "d"}

RQ4_HINDI_MCQ = {
    "ए": "a",
    "बी": "b",
    "सी": "c",
    "डी": "d",
}

RQ4_HINDI_YES_NO = {
    "हाँ": "yes",
    "हां": "yes",
    "नहीं": "no",
    "नही": "no",
}


@dataclass(frozen=True)
class RQ4Evaluation:
    predicted: str | None
    status: str  # correct | incorrect | unparseable
    score: float


def _clean_output(model_output: str) -> str:
    cleaned = str(model_output or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:text|txt)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def _explicit_marker_answer(cleaned: str, gold_clean: str) -> str | None:
    """Extract an explicitly marked answer or a complete single option line."""
    if gold_clean in RQ4_MCQ:
        marker = re.findall(
            r"(?:the\s+correct\s+answer\s+is|correct\s+answer|final\s+answer|answer)"
            r"\s*[:\-]?\s*\(?\s*([ABCD])\s*\)?\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if marker:
            return marker[-1].lower()

        hindi_marker = re.findall(
            r"(?:उत्तर|जवाब)\s*[:\-]?\s*\(?\s*(ए|बी|सी|डी)\s*\)?",
            cleaned,
        )
        if hindi_marker:
            return RQ4_HINDI_MCQ[hindi_marker[-1]]

        # Accept a complete option such as "A. Entailment".
        # fullmatch prevents arbitrary A/B/C/D inside reasoning from
        # being interpreted as the answer.
        option_line = re.fullmatch(
            r"\s*\(?\s*([ABCD])\s*[.)\-:]\s*.+?\s*",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if option_line:
            return option_line.group(1).lower()

        hindi_option_line = re.fullmatch(
            r"\s*\(?\s*(ए|बी|सी|डी)\s*[.)\-:]\s*.+?\s*",
            cleaned,
            flags=re.DOTALL,
        )
        if hindi_option_line:
            return RQ4_HINDI_MCQ[hindi_option_line.group(1)]

    if gold_clean in RQ4_YES_NO:
        marker = re.findall(
            r"(?:answer|final\s+answer|correct\s+answer|उत्तर|जवाब)"
            r"\s*[:\-]?\s*(yes|no|हाँ|हां|नहीं|नही)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if marker:
            token = marker[-1].lower()
            return RQ4_HINDI_YES_NO.get(token, token)

    return None


def extract_rq4_answer(model_output: str, gold: str) -> str | None:
    """Strict RQ4 answer extraction.

    Accepted MCQ forms:
      - bare labels: ``B`` / ``B.`` / ``(B)``
      - explicit markers: ``Answer: B`` / ``Final answer: B``
      - complete option lines: ``A. Entailment``

    A/B/C/D appearing arbitrarily inside a reasoning trace is never treated
    as an answer unless the response explicitly marks it or the entire
    response is a single option line.
    """
    cleaned = _clean_output(model_output)
    gold_clean = str(gold).strip().lower()

    if not cleaned:
        return None

    explicit = _explicit_marker_answer(cleaned, gold_clean)
    if explicit is not None:
        return explicit

    if gold_clean in RQ4_MCQ:
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
        if bare_number:
            return bare_number.group(1)

    return None


def evaluate_rq4(model_output: str, gold: str) -> RQ4Evaluation:
    predicted = extract_rq4_answer(model_output, gold)
    gold_clean = str(gold).strip().lower()

    if predicted is None:
        return RQ4Evaluation(None, "unparseable", 0.0)

    if predicted == gold_clean:
        return RQ4Evaluation(predicted, "correct", 1.0)

    return RQ4Evaluation(predicted, "incorrect", 0.0)
