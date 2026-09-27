import re

from src.benchmarks.loader import BenchmarkItem, register

GOLD_ANSWER_RE = re.compile(r"####\s*([\-0-9,.]+)")

MATH_PROMPT_TEMPLATE = (
    "निम्नलिखित गणित की समस्या को चरण दर चरण हल करें। अंतिम उत्तर '#### ' के बाद लिखें।\n\n"
    "प्रश्न: {question}\n\nउत्तर:"
)


def _extract_gold_number(answer_field: str) -> str | None:
    match = GOLD_ANSWER_RE.search(answer_field)
    if not match:
        return None
    return match.group(1).replace(",", "")


@register("gsm8k_hi")
def load_gsm8k_hi() -> list[BenchmarkItem]:
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset

    local_path = hf_hub_download("nvidia/GSM8K-Hi", "test.jsonl", repo_type="dataset")
    raw = load_dataset("json", data_files=local_path)["train"]

    items = []
    dropped = 0
    for i, row in enumerate(raw):
        gold = _extract_gold_number(row["answer"])
        if gold is None:
            dropped += 1
            continue
        prompt = MATH_PROMPT_TEMPLATE.format(question=row["question"])
        items.append(
            BenchmarkItem(
                benchmark="gsm8k_hi",
                item_id=f"gsm8k_hi_{i:04d}",
                language="hi",
                task_type="generation",
                input={"question": row["question"]},
                gold=gold,
                prompt_template=prompt,
            )
        )

    if dropped:
        print(f"gsm8k_hi: dropped {dropped} item(s) with unparseable gold answer")

    return items