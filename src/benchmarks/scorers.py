import collections
import json
import re

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning_trace(model_output: str) -> str:
    return THINK_BLOCK_RE.sub("", model_output).strip()


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def extract_sentiment_label(model_output: str) -> str | None:
    cleaned = strip_reasoning_trace(model_output)
    normalized = normalize_label(cleaned)

    has_positive = "positive" in normalized or "सकारात्मक" in cleaned
    has_negative = "negative" in normalized or "नकारात्मक" in cleaned

    if has_positive and not has_negative:
        return "Positive"
    if has_negative and not has_positive:
        return "Negative"
    return None


def score_sentiment_accuracy(model_output: str, gold: str) -> float:
    predicted = extract_sentiment_label(model_output)
    if predicted is None:
        return 0.0
    return float(predicted == gold)


SCORERS = {
    "indicsentiment_hi": score_sentiment_accuracy,
    "indicsentiment_bn": score_sentiment_accuracy,
}


NLI_LABEL_KEYWORDS = {
    "entailment": ["entailment", "समावेशन", "अनुसरण", "निहितार्थ", "निहितता", "निहित", "एंटेलमेंट"],
    "neutral": ["neutral", "तटस्थ", "न्यूट्रल"],
    "contradiction": ["contradiction", "विरोधाभास", "विरोध", "कॉन्ट्राडिक्शन", "कॉन्ट्रैडिक्शन"],
}


def extract_nli_label(model_output: str) -> str | None:
    cleaned = strip_reasoning_trace(model_output)
    normalized = normalize_label(cleaned)

    matches = set()
    for label, keywords in NLI_LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw.isascii():
                if normalize_label(kw) in normalized:
                    matches.add(label)
                    break
            elif kw in cleaned:
                matches.add(label)
                break

    if len(matches) == 1:
        return matches.pop()
    return None


def score_nli_accuracy(model_output: str, gold: str) -> float:
    predicted = extract_nli_label(model_output)
    if predicted is None:
        return 0.0
    return float(predicted == gold)


SCORERS["indicxnli_hi"] = score_nli_accuracy
SCORERS["indicxnli_bn"] = score_nli_accuracy


GOLD_MARKER_RE = re.compile(r"####\s*([\-0-9,.]+)")
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(model_output: str) -> str | None:
    cleaned = strip_reasoning_trace(model_output)

    marker_match = GOLD_MARKER_RE.search(cleaned)
    if marker_match:
        return marker_match.group(1).replace(",", "")

    numbers = NUMBER_RE.findall(cleaned)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def score_math_exact_match(model_output: str, gold: str) -> float:
    predicted = extract_final_number(model_output)
    if predicted is None:
        return 0.0
    try:
        return float(float(predicted) == float(gold))
    except ValueError:
        return 0.0


SCORERS["gsm8k_hi"] = score_math_exact_match


def extract_function_call_json(model_output: str) -> dict | None:
    cleaned = strip_reasoning_trace(model_output).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    cleaned = cleaned.replace("\\_", "_")

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return normalize_function_call(parsed)


NAME_KEYS = ("name", "function_name", "function")
ARGS_KEYS = ("arguments", "parameters", "params", "args")


def normalize_function_call(parsed: dict) -> dict | None:
    name_key = next((k for k in NAME_KEYS if k in parsed), None)
    args_key = next((k for k in ARGS_KEYS if k in parsed), None)

    if name_key is not None and args_key is not None and isinstance(parsed[name_key], str):
        args_value = parsed[args_key]
        if not isinstance(args_value, dict):
            return None
        return {parsed[name_key]: args_value}

    return parsed


def score_bfcl(model_output: str, function_schema: dict, ground_truth: dict) -> float:
    from src.benchmarks.bfcl_lib.ast_checker import simple_function_checker
    from src.benchmarks.bfcl_lib.enums import Language

    parsed_call = extract_function_call_json(model_output)
    if parsed_call is None:
        return 0.0

    result = simple_function_checker(
        func_description=function_schema,
        model_output=parsed_call,
        possible_answer=ground_truth,
        language=Language.PYTHON,
        model_name="generic",
    )
    return float(result["valid"])


OPTION_NUMBER_RE = re.compile(r"[1234]")
OPTION_MARKER_RE = re.compile(r"(?:उत्तर|উত্তর|answer)\s*[:\-]?\s*([1234])", re.IGNORECASE)


def extract_mcq_option(model_output: str) -> str | None:
    cleaned = strip_reasoning_trace(model_output)

    marker_matches = OPTION_MARKER_RE.findall(cleaned)
    if marker_matches:
        return marker_matches[-1]

    matches = OPTION_NUMBER_RE.findall(cleaned)
    if not matches:
        return None
    return matches[-1]


def score_mcq_accuracy(model_output: str, gold: str) -> float:
    predicted = extract_mcq_option(model_output)
    if predicted is None:
        return 0.0
    return float(predicted == gold)


SCORERS["milu_hi"] = score_mcq_accuracy
SCORERS["milu_bn"] = score_mcq_accuracy


def _tokenize_for_f1(text: str) -> list[str]:
    cleaned = re.sub(r"[.,!?\"'।]", " ", text.lower())
    return cleaned.split()


def _f1_single(prediction_tokens: list[str], reference_tokens: list[str]) -> float:
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    common = collections.Counter(prediction_tokens) & collections.Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_f1_max_over_references(model_output: str, reference_answers: list[str]) -> float:
    cleaned = strip_reasoning_trace(model_output)
    prediction_tokens = _tokenize_for_f1(cleaned)

    scores = [_f1_single(prediction_tokens, _tokenize_for_f1(ref)) for ref in reference_answers]
    return max(scores) if scores else 0.0


SCORERS["chatrag_hi"] = score_f1_max_over_references


def score_ifeval(model_output: str, instruction_id_list: list, kwargs_list: list) -> float:
    from src.benchmarks.ifeval_lib import instructions_registry

    cleaned = strip_reasoning_trace(model_output)
    is_following_list = []

    for instruction_id, raw_kwargs in zip(instruction_id_list, kwargs_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        filtered_kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}
        instruction.build_description(**filtered_kwargs)

        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=model_output)

        is_following_list.append(bool(cleaned.strip()) and instruction.check_following(cleaned))

    return float(all(is_following_list))


JUDGE_PROMPT_TEMPLATE = """

Evaluate the quality of the model's responses to questions from a benchmark dataset on a scale of 1-5 (score can be a decimal fraction format number) across the following parameters:

Factual Accuracy: Given an input question, ground truth facts relevant to the question, and the model/bot's answer, evaluate how well the information in the model's answer aligns with the provided ground truth facts. Assign a score on a scale of 1 to 5 based on the following criteria: a score of 5 indicates complete alignment with all ground truth facts; a score of 3 represents partial alignment where approximately half of the facts are correct; and a score of 1 denotes complete misalignment with the ground truth facts. Scores between these benchmarks can reflect varying degrees of alignment or discrepancies.

Relevance: Assess how well the model's answer directly addresses the question. A score of 5 indicates a highly relevant answer, while a score of 1 indicates an irrelevant or off-topic response.

Clarity: Evaluate the clarity and coherence of the model's answer. A score of 5 means the answer is well-structured and easy to understand, while a score of 1 means it is confusing or poorly constructed.

Language Consistency: Ensure that the language of the response matches the language of the question unless otherwise specified. Penalize cases where there is a mismatch between the input language specified in the question and the response language.

Conciseness: Rate how concise the answer is while still providing necessary information. A score of 5 indicates the answer is succinct and to the point, while a score of 1 indicates excessive verbosity or unnecessary information.

Input Details:

Question: {question}
Ground Truth Facts: {ground_truth}
Model/Bot Answer: {model_answer}
After evaluating each parameter, provide an overall rating on a scale of 1-5 considering all the parameters. The parameter factual accuracy should have more weightage in the overall score.

Output Format:
Return the evaluation scores in the following JSON format(Return only the JSON and nothing else):
{{
  "Factual Accuracy": score,
  "Relevance": score,
  "Clarity": score,
  "Language Consistency": score,
  "Conciseness": score,
  "Overall": average_score
}}
"""


def score_llm_judge(question: str, ground_truth: str, model_answer: str, judge_client) -> float:
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, ground_truth=ground_truth, model_answer=model_answer)
    judge_output = judge_client.generate(prompt, temperature=0.0, max_new_tokens=300)
    cleaned = strip_reasoning_trace(judge_output).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
        overall = float(parsed["Overall"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return 0.0

    return max(0.0, min(1.0, overall / 5.0))


def score_item(benchmark: str, model_output: str, gold) -> float:
    if benchmark not in SCORERS:
        raise ValueError(f"no scorer registered for benchmark '{benchmark}'")
    return SCORERS[benchmark](model_output, gold)