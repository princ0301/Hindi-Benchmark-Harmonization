import json

from src.benchmarks.loader import BenchmarkItem, register

BFCL_PROMPT_TEMPLATE = """निम्नलिखित उपयोगकर्ता अनुरोध के लिए दिए गए फ़ंक्शन को कॉल करें। केवल एक JSON ऑब्जेक्ट लिखें, कोई अन्य पाठ नहीं। प्रारूप इस प्रकार होना चाहिए: {{"{function_name}": {{"param1": value1, "param2": value2}}}}

फ़ंक्शन विवरण: {function_schema}

उपयोगकर्ता अनुरोध: {question}

JSON:"""


@register("bfcl_hi")
def load_bfcl_hi() -> list[BenchmarkItem]:
    from huggingface_hub import hf_hub_download

    q_path = hf_hub_download("nvidia/BFCL-Hi", "BFCL_v2_live_simple.json", repo_type="dataset")
    a_path = hf_hub_download("nvidia/BFCL-Hi", "possible_answer/BFCL_v2_live_simple.json", repo_type="dataset")

    with open(q_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    with open(a_path, encoding="utf-8") as f:
        answers = {row["id"]: row for row in (json.loads(line) for line in f)}

    items = []
    dropped = 0
    for row in questions:
        answer_row = answers.get(row["id"])
        if answer_row is None or len(row["function"]) != 1:
            dropped += 1
            continue

        function_schema = row["function"][0]
        ground_truth = answer_row["ground_truth"][0]
        user_question = row["question"][0]["content"]

        prompt = BFCL_PROMPT_TEMPLATE.format(
            function_name=function_schema["name"],
            function_schema=json.dumps(function_schema, ensure_ascii=False),
            question=user_question,
        )

        items.append(
            BenchmarkItem(
                benchmark="bfcl_hi",
                item_id=row["id"],
                language="hi",
                task_type="generation",
                input={"function_schema": function_schema, "ground_truth": ground_truth},
                gold=None,
                prompt_template=prompt,
            )
        )

    if dropped:
        print(f"bfcl_hi: dropped {dropped} item(s) with missing ground truth or non-single function")

    return items