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

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    cleaned = re.sub(
        r"^```(?:text|txt)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    return cleaned


def _explicit_marker_answer(cleaned: str, gold_clean: str) -> str | None:
    """
    Extract answers when the model explicitly identifies the answer.

    Examples:
        Answer: B
        Final answer: C
        The correct answer is A
        उत्तर: बी

    Also supports MCQ option-list style outputs:

        A. Entailment
        B. Neutral
        C. Contradiction

    The option-list parser is deliberately constrained so that arbitrary
    A/B/C/D letters inside reasoning are never treated as answers.
    """

    if gold_clean in RQ4_MCQ:

        # ---------------------------------------------------------------
        # 1. Explicit English answer markers
        # ---------------------------------------------------------------
        marker = re.findall(
            r"""
            (?:
                the\s+correct\s+answer\s+is
                |
                correct\s+answer
                |
                final\s+answer
                |
                answer
            )
            \s*[:\-]?\s*
            \(?\s*([ABCD])\s*\)?
            \b
            """,
            cleaned,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        if marker:
            return marker[-1].lower()

        # ---------------------------------------------------------------
        # 2. Explicit Hindi answer markers
        # ---------------------------------------------------------------
        hindi_marker = re.findall(
            r"""
            (?:
                उत्तर
                |
                जवाब
            )
            \s*[:\-]?\s*
            \(?\s*(ए|बी|सी|डी)\s*\)?
            \b
            """,
            cleaned,
            flags=re.VERBOSE,
        )

        if hindi_marker:
            return RQ4_HINDI_MCQ[hindi_marker[-1]]

        # ---------------------------------------------------------------
        # 3. Option-label + option-text
        #
        # Examples:
        #   A. Entailment
        #   B. Neutral
        #   C. Contradiction
        #
        # IMPORTANT:
        # We only accept this when the entire response is effectively
        # one option line. This prevents arbitrary A/B/C/D in reasoning
        # from being interpreted as the answer.
        # ---------------------------------------------------------------
        option_line = re.fullmatch(
            r"""
            \s*
            \(?\s*([ABCD])\s*[\.\):\-]\s*
            .+?
            \s*
            """,
            cleaned,
            flags=re.IGNORECASE | re.VERBOSE | re.DOTALL,
        )

        if option_line:
            return option_line.group(1).lower()

        # Hindi option labels
        hindi_option_line = re.fullmatch(
            r"""
            \s*
            \(?\s*(ए|बी|सी|डी)\s*[\.\):\-]\s*
            .+?
            \s*
            """,
            cleaned,
            flags=re.VERBOSE | re.DOTALL,
        )

        if hindi_option_line:
            return RQ4_HINDI_MCQ[hindi_option_line.group(1)]

    # -------------------------------------------------------------------
    # YES / NO
    # -------------------------------------------------------------------
    if gold_clean in RQ4_YES_NO:

        marker = re.findall(
            r"""
            (?:
                answer
                |
                final\s+answer
                |
                correct\s+answer
                |
                उत्तर
                |
                जवाब
            )
            \s*[:\-]?\s*
            (yes|no|हाँ|हां|नहीं|नही)
            \b
            """,
            cleaned,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        if marker:
            token = marker[-1].lower()

            for hindi, english in RQ4_HINDI_YES_NO.items():
                if token == hindi.lower():
                    return english

            return token

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
    """
    Strict RQ4 extraction.

    Accepted MCQ forms include bare labels (`B`), leading option labels with
    option text (`B. Entailment`), and explicit answer markers (`Answer: B`).
    We never infer an answer from arbitrary letters appearing later in a
    verbose reasoning trace.
    Accepted:
        B
        B.
        (B)
        Answer: B
        Final answer: B
        The correct answer is B
        A. Entailment
        C. Contradiction

    Rejected:
        arbitrary A/B/C/D occurring inside reasoning.

    For MCQ responses, an answer is only extracted when it is either:
      1. explicitly marked as an answer, or
      2. the entire response is a single option line.
    """

    cleaned = _clean_output(model_output)
    gold_clean = str(gold).strip().lower()

    if not cleaned:
        return None

    # First try explicit/structured answers.
    explicit = _explicit_marker_answer(cleaned, gold_clean)

    if explicit is not None:
        return explicit

    # -------------------------------------------------------------------
    # Bare MCQ answer
    # -------------------------------------------------------------------
    if gold_clean in RQ4_MCQ:
        leading = _leading_option_answer(cleaned)
        if leading is not None:
            return leading

        bare = re.fullmatch(r"\(?\s*([ABCD])\s*[.)]?\s*", cleaned, flags=re.IGNORECASE)

        bare = re.fullmatch(
            r"\(?\s*([ABCD])\s*[.)]?\s*",
            cleaned,
            flags=re.IGNORECASE,
        )

        if bare:
            return bare.group(1).lower()

        bare_hindi = re.fullmatch(
            r"\(?\s*(ए|बी|सी|डी)\s*[.)]?\s*",
            cleaned,
        )

        if bare_hindi:
            return RQ4_HINDI_MCQ[bare_hindi.group(1)]

        return None

    # -------------------------------------------------------------------
    # Bare YES / NO
    # -------------------------------------------------------------------
    if gold_clean in RQ4_YES_NO:

        bare = re.fullmatch(
            r"\s*(yes|no)\s*[.!]?\s*",
            cleaned,
            flags=re.IGNORECASE,
        )

        if bare:
            return bare.group(1).lower()

        bare_hindi = re.fullmatch(
            r"\s*(हाँ|हां|नहीं|नही)\s*[.!]?\s*",
            cleaned,
        )

        if bare_hindi:
            return RQ4_HINDI_YES_NO[bare_hindi.group(1)]

        return None

    # -------------------------------------------------------------------
    # Numeric answers
    # -------------------------------------------------------------------
    if re.fullmatch(r"\d+", gold_clean):

        bare_number = re.fullmatch(
            r"\s*(\d+)\s*[.]?\s*",
            cleaned,
        )

        if bare_number:
            return bare_number.group(1)

        return None

    return None


def evaluate_rq4(model_output: str, gold: str) -> RQ4Evaluation:
    predicted = extract_rq4_answer(model_output, gold)
    gold_clean = str(gold).strip().lower()

    if predicted is None:
        return RQ4Evaluation(
            None,
            "unparseable",
            0.0,
        )

    if predicted == gold_clean:
        return RQ4Evaluation(
            predicted,
            "correct",
            1.0,
        )

    return RQ4Evaluation(
        predicted,
        "incorrect",
        0.0,
    )