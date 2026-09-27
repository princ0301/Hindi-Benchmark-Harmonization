import json

from src.benchmarks.scorers import (
    extract_final_number,
    extract_function_call_json,
    extract_mcq_option,
    extract_nli_label,
    extract_sentiment_label,
    score_bfcl,
    score_f1_max_over_references,
    score_ifeval,
    score_llm_judge,
    score_math_exact_match,
    score_mcq_accuracy,
    score_nli_accuracy,
    score_sentiment_accuracy,
    strip_reasoning_trace,
)


class FakeJudgeClient:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 300) -> str:
        return self._response_text


def test_strip_reasoning_trace_removes_think_block():
    text = "<think>यह एक नकारात्मक समीक्षा है</think>Positive"
    assert strip_reasoning_trace(text) == "Positive"


def test_strip_reasoning_trace_no_block_unchanged():
    text = "Positive"
    assert strip_reasoning_trace(text) == "Positive"


def test_extract_sentiment_label_plain_positive():
    assert extract_sentiment_label("Positive") == "Positive"


def test_extract_sentiment_label_plain_negative():
    assert extract_sentiment_label("Negative") == "Negative"


def test_extract_sentiment_label_with_surrounding_text():
    text = "इस समीक्षा की भावना है: Negative"
    assert extract_sentiment_label(text) == "Negative"


def test_extract_sentiment_label_lowercase():
    assert extract_sentiment_label("positive") == "Positive"


def test_extract_sentiment_label_with_reasoning_trace():
    text = "<think>the review sounds negative</think>Positive"
    assert extract_sentiment_label(text) == "Positive"


def test_extract_sentiment_label_both_words_present_is_ambiguous():
    text = "It is not negative, it is positive"
    assert extract_sentiment_label(text) is None


def test_extract_sentiment_label_neither_word_present():
    text = "यह एक अच्छी समीक्षा है"
    assert extract_sentiment_label(text) is None


def test_extract_sentiment_label_hindi_positive():
    assert extract_sentiment_label("सकारात्मक") == "Positive"


def test_extract_sentiment_label_hindi_negative():
    assert extract_sentiment_label("नकारात्मक") == "Negative"


def test_extract_sentiment_label_hindi_both_words_present_is_ambiguous():
    text = "यह सकारात्मक नहीं बल्कि नकारात्मक है"
    assert extract_sentiment_label(text) is None


def test_score_sentiment_accuracy_correct():
    assert score_sentiment_accuracy("Positive", "Positive") == 1.0


def test_score_sentiment_accuracy_incorrect():
    assert score_sentiment_accuracy("Positive", "Negative") == 0.0


def test_score_sentiment_accuracy_unparseable_output_scored_wrong():
    assert score_sentiment_accuracy("मुझे नहीं पता", "Positive") == 0.0


def test_extract_nli_label_plain_english():
    assert extract_nli_label("entailment") == "entailment"
    assert extract_nli_label("neutral") == "neutral"
    assert extract_nli_label("contradiction") == "contradiction"


def test_extract_nli_label_hindi():
    assert extract_nli_label("विरोधाभास") == "contradiction"
    assert extract_nli_label("तटस्थ") == "neutral"


def test_extract_nli_label_hindi_entailment_synonym():
    assert extract_nli_label("निहितता") == "entailment"
    assert extract_nli_label("निहित हैं") == "entailment"


def test_extract_nli_label_transliterated_english():
    assert extract_nli_label("एंटेलमेंट") == "entailment"
    assert extract_nli_label("न्यूट्रल") == "neutral"
    assert extract_nli_label("कॉन्ट्राडिक्शन") == "contradiction"


def test_extract_nli_label_unrelated_hindi_word_not_matched():
    assert extract_nli_label("परिकल्पना") is None
    assert extract_nli_label("समन्वय") is None


def test_extract_nli_label_with_surrounding_text():
    text = "इन दोनों वाक्यों के बीच संबंध है: contradiction"
    assert extract_nli_label(text) == "contradiction"


def test_extract_nli_label_multiple_matches_is_ambiguous():
    text = "यह न तो entailment है और न ही contradiction"
    assert extract_nli_label(text) is None


def test_extract_nli_label_neither_present():
    assert extract_nli_label("मुझे नहीं पता") is None


def test_score_nli_accuracy_correct():
    assert score_nli_accuracy("neutral", "neutral") == 1.0


def test_score_nli_accuracy_incorrect():
    assert score_nli_accuracy("entailment", "contradiction") == 0.0


def test_extract_final_number_with_marker():
    text = "गणना: 16 - 3 - 4 = 9\n\n9 * 2 = 18\n\n#### 18"
    assert extract_final_number(text) == "18"


def test_extract_final_number_with_comma():
    text = "#### 1,234"
    assert extract_final_number(text) == "1234"


def test_extract_final_number_no_marker_falls_back_to_last_number():
    text = "पहले हमें 9 अंडे मिलते हैं, फिर 9 * 2 = 18 डॉलर"
    assert extract_final_number(text) == "18"


def test_extract_final_number_with_reasoning_trace():
    text = "<think>let me compute 9 * 2 = 18</think>#### 18"
    assert extract_final_number(text) == "18"


def test_extract_final_number_no_number_present():
    assert extract_final_number("मुझे नहीं पता") is None


def test_score_math_exact_match_correct():
    assert score_math_exact_match("#### 18", "18") == 1.0


def test_score_math_exact_match_decimal_equivalence():
    assert score_math_exact_match("#### 18.0", "18") == 1.0


def test_score_math_exact_match_incorrect():
    assert score_math_exact_match("#### 20", "18") == 0.0


def test_score_math_exact_match_unparseable_scored_wrong():
    assert score_math_exact_match("मुझे नहीं पता", "18") == 0.0


def test_extract_mcq_option_plain():
    assert extract_mcq_option("2") == "2"


def test_extract_mcq_option_with_explanatory_preamble():
    text = "विकल्प 1 गलत है, विकल्प 3 भी गलत है, सही उत्तर विकल्प 2 है।\n\nउत्तर: 2"
    assert extract_mcq_option(text) == "2"


def test_extract_mcq_option_no_number_present():
    assert extract_mcq_option("मुझे नहीं पता") is None


def test_extract_mcq_option_marker_at_start_then_confusing_numbers():
    text = (
        "उत्तर: 2\n\n"
        "H = (0.7) / (4 x 10^-7)\n"
        "H ~ 1671 AT/m\n"
        "AT = 1671 * 0.003\n"
        "AT ~ 5 AT"
    )
    assert extract_mcq_option(text) == "2"


def test_extract_mcq_option_marker_at_end_after_reasoning():
    text = "12 + 1 + 16 + 20 = 49, इसलिए सही उत्तर विकल्प 3 है।\n\nउत्तर: 3"
    assert extract_mcq_option(text) == "3"


def test_extract_mcq_option_bengali_marker():
    text = "সঠিক উত্তর হলো ৩ নম্বর বিকল্প।\n\nউত্তর: 3"
    assert extract_mcq_option(text) == "3"


def test_extract_mcq_option_no_marker_falls_back_to_bare_digit():
    assert extract_mcq_option("4") == "4"


def test_score_mcq_accuracy_correct():
    assert score_mcq_accuracy("3", "3") == 1.0


def test_score_mcq_accuracy_incorrect():
    assert score_mcq_accuracy("1", "4") == 0.0


REAL_CHEESE_REFERENCES = [
    "पनीर बनाने के लिए गाय के दूध के अलावा बकरी और भेड़ के दूध का भी उपयोग किया जाता है।",
    "गाय के दूध के अलावा, भैंस, बकरी और भेड़ के दूध का उपयोग पनीर बनाने में किया जाता है।",
]


def test_score_f1_exact_match_scores_one():
    assert score_f1_max_over_references(REAL_CHEESE_REFERENCES[0], REAL_CHEESE_REFERENCES) == 1.0


def test_score_f1_partial_overlap_scores_between_zero_and_one():
    prediction = "बकरी और भेड़ के दूध का भी उपयोग किया जाता है।"
    score = score_f1_max_over_references(prediction, REAL_CHEESE_REFERENCES)
    assert 0.0 < score < 1.0


def test_score_f1_completely_unrelated_scores_near_zero():
    prediction = "आज मौसम बहुत अच्छा है।"
    score = score_f1_max_over_references(prediction, REAL_CHEESE_REFERENCES)
    assert score < 0.1


def test_score_f1_empty_prediction_scores_zero():
    assert score_f1_max_over_references("", REAL_CHEESE_REFERENCES) == 0.0


def test_score_f1_takes_max_across_multiple_references():
    prediction = "गाय के दूध के अलावा, भैंस, बकरी और भेड़ के दूध का उपयोग पनीर बनाने में किया जाता है।"
    score = score_f1_max_over_references(prediction, REAL_CHEESE_REFERENCES)
    assert score == 1.0


def test_score_llm_judge_plain_json():
    judge_response = '{"Factual Accuracy": 5, "Relevance": 5, "Clarity": 4, "Language Consistency": 5, "Conciseness": 4, "Overall": 4.5}'
    client = FakeJudgeClient(judge_response)
    assert score_llm_judge("q", "gt", "answer", client) == 0.9


def test_score_llm_judge_json_in_code_fence():
    judge_response = '```json\n{"Overall": 2.5}\n```'
    client = FakeJudgeClient(judge_response)
    assert score_llm_judge("q", "gt", "answer", client) == 0.5


def test_score_llm_judge_malformed_output_scored_zero():
    client = FakeJudgeClient("the model did okay, I'd say around 4/5")
    assert score_llm_judge("q", "gt", "answer", client) == 0.0


def test_score_llm_judge_score_clamped_to_valid_range():
    judge_response = '{"Overall": 7}'
    client = FakeJudgeClient(judge_response)
    assert score_llm_judge("q", "gt", "answer", client) == 1.0


def test_score_ifeval_letter_frequency_pass():
    instruction_id_list = ["keywords:letter_frequency"]
    kwargs_list = [{"letter": "श", "let_frequency": 3, "let_relation": "at least"}]
    response = "शिक्षा में सुधार शीघ्र होना चाहिए, यह शासन की जिम्मेदारी है।"
    assert score_ifeval(response, instruction_id_list, kwargs_list) == 1.0


def test_score_ifeval_letter_frequency_fail():
    instruction_id_list = ["keywords:letter_frequency"]
    kwargs_list = [{"letter": "श", "let_frequency": 10, "let_relation": "at least"}]
    response = "शिक्षा में सुधार होना चाहिए।"
    assert score_ifeval(response, instruction_id_list, kwargs_list) == 0.0


def test_score_ifeval_empty_response_fails():
    instruction_id_list = ["keywords:letter_frequency"]
    kwargs_list = [{"letter": "श", "let_frequency": 1, "let_relation": "at least"}]
    assert score_ifeval("", instruction_id_list, kwargs_list) == 0.0


def test_score_ifeval_multiple_instructions_all_must_pass():
    instruction_id_list = ["keywords:letter_frequency", "punctuation:no_comma"]
    kwargs_list = [
        {"letter": "श", "let_frequency": 1, "let_relation": "at least"},
        {},
    ]
    response_with_comma = "शिक्षा, सुधार आवश्यक है।"
    assert score_ifeval(response_with_comma, instruction_id_list, kwargs_list) == 0.0

    response_without_comma = "शिक्षा में सुधार आवश्यक है।"
    assert score_ifeval(response_without_comma, instruction_id_list, kwargs_list) == 1.0


BFCL_FUNCTION_SCHEMA = {
    "name": "get_user_info",
    "description": "Retrieve details for a specific user by their unique identifier.",
    "parameters": {
        "type": "dict",
        "required": ["user_id"],
        "properties": {
            "user_id": {"type": "integer", "description": "The unique identifier of the user."},
            "special": {"type": "string", "description": "Special info.", "default": "none"},
        },
    },
}
BFCL_GROUND_TRUTH = {"get_user_info": {"user_id": [7890], "special": ["black"]}}


def test_extract_function_call_json_plain():
    text = '{"get_user_info": {"user_id": 7890, "special": "black"}}'
    assert extract_function_call_json(text) == {"get_user_info": {"user_id": 7890, "special": "black"}}


def test_extract_function_call_json_in_code_fence():
    text = '```json\n{"get_user_info": {"user_id": 7890}}\n```'
    assert extract_function_call_json(text) == {"get_user_info": {"user_id": 7890}}


def test_extract_function_call_json_malformed_returns_none():
    assert extract_function_call_json("I would call get_user_info with id 7890") is None


def test_extract_function_call_json_normalizes_openai_style_name_arguments():
    text = '{"name": "get_user_info", "arguments": {"user_id": 7890, "special": "black"}}'
    assert extract_function_call_json(text) == {"get_user_info": {"user_id": 7890, "special": "black"}}


def test_extract_function_call_json_normalizes_function_name_parameters_style():
    text = '{"function_name": "get_user_info", "parameters": {"user_id": 7890}}'
    assert extract_function_call_json(text) == {"get_user_info": {"user_id": 7890}}


def test_extract_function_call_json_canonical_format_unchanged():
    text = '{"get_user_info": {"user_id": 7890}}'
    assert extract_function_call_json(text) == {"get_user_info": {"user_id": 7890}}


def test_extract_function_call_json_handles_escaped_underscores():
    text = '{"function\\_name": "get\\_user\\_info", "parameters": {"user\\_id": "7890"}}'
    assert extract_function_call_json(text) == {"get_user_info": {"user_id": "7890"}}


def test_score_bfcl_correct_call_with_openai_style_wrapper():
    output = '{"function_name": "get_user_info", "parameters": {"user_id": 7890, "special": "black"}}'
    assert score_bfcl(output, BFCL_FUNCTION_SCHEMA, BFCL_GROUND_TRUTH) == 1.0


def test_score_bfcl_string_id_against_integer_ground_truth_is_strict():
    # Documents official, unmodified BFCL behavior: a quoted numeric value ("7890")
    # does not match an unquoted integer ground truth (7890), even though the
    # represented value is identical. See bfcl_lib/NOTICE.md for discussion.
    output = '{"function_name": "get_user_info", "parameters": {"user_id": "7890", "special": "black"}}'
    assert score_bfcl(output, BFCL_FUNCTION_SCHEMA, BFCL_GROUND_TRUTH) == 0.0


def test_score_bfcl_schema_echoed_back_instead_of_values_scored_wrong():
    output = json.dumps({"function_name": "get_user_info", "parameters": BFCL_FUNCTION_SCHEMA["parameters"]})
    assert score_bfcl(output, BFCL_FUNCTION_SCHEMA, BFCL_GROUND_TRUTH) == 0.0


def test_score_bfcl_correct_call():
    output = '{"get_user_info": {"user_id": 7890, "special": "black"}}'
    assert score_bfcl(output, BFCL_FUNCTION_SCHEMA, BFCL_GROUND_TRUTH) == 1.0


def test_score_bfcl_wrong_parameter_value():
    output = '{"get_user_info": {"user_id": 1111, "special": "black"}}'
    assert score_bfcl(output, BFCL_FUNCTION_SCHEMA, BFCL_GROUND_TRUTH) == 0.0


def test_score_bfcl_missing_required_parameter():
    output = '{"get_user_info": {"special": "black"}}'
    assert score_bfcl(output, BFCL_FUNCTION_SCHEMA, BFCL_GROUND_TRUTH) == 0.0


def test_score_bfcl_unparseable_output_scored_wrong():
    output = "मुझे यह जानकारी नहीं मिल सकी।"
    assert score_bfcl(output, BFCL_FUNCTION_SCHEMA, BFCL_GROUND_TRUTH) == 0.0


def test_score_ifeval_paragraph_first_word_trailing_whitespace_in_kwargs_does_not_break_match():
    instruction_id_list = ["length_constraints:nth_paragraph_first_word"]
    kwargs_list = [{"first_word": "लोक ", "nth_paragraph": 1, "num_paragraphs": 2}]
    response = "लोक भारतीय सिनेमा की नींव है।\n\nदूसरा पैराग्राफ यहाँ है।"
    assert score_ifeval(response, instruction_id_list, kwargs_list) == 1.0


def test_score_ifeval_paragraph_first_word_matches_ignoring_markdown_bold():
    instruction_id_list = ["length_constraints:nth_paragraph_first_word"]
    kwargs_list = [{"first_word": "नमस्ते", "nth_paragraph": 1, "num_paragraphs": 2}]
    response = "**नमस्ते** यह पहला पैराग्राफ है।\n\nदूसरा पैराग्राफ यहाँ है।"
    assert score_ifeval(response, instruction_id_list, kwargs_list) == 1.0