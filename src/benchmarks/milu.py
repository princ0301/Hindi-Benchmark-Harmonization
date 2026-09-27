import os

from src.benchmarks.loader import BenchmarkItem, register

PROMPT_TEMPLATES = {
    "hi": (
        "निम्नलिखित प्रश्न का सही उत्तर चुनें। केवल विकल्प संख्या (1, 2, 3, या 4) लिखें।\n\n"
        "प्रश्न: {question}\n"
        "1. {option1}\n2. {option2}\n3. {option3}\n4. {option4}\n\nउत्तर:"
    ),
    "bn": (
        "নিম্নলিখিত প্রশ্নের সঠিক উত্তর বেছে নিন। শুধুমাত্র বিকল্প সংখ্যা (1, 2, 3, বা 4) লিখুন।\n\n"
        "প্রশ্ন: {question}\n"
        "1. {option1}\n2. {option2}\n3. {option3}\n4. {option4}\n\nউত্তর:"
    ),
}

LANGUAGE_TO_MILU_DIR = {"hi": "Hindi", "bn": "Bengali"}

TARGET_TO_NUMBER = {"option1": "1", "option2": "2", "option3": "3", "option4": "4"}


def _load_milu(language: str) -> list[BenchmarkItem]:
    from datasets import load_dataset

    benchmark_name = f"milu_{language}"
    milu_dir = LANGUAGE_TO_MILU_DIR[language]
    raw = load_dataset("ai4bharat/MILU", data_dir=milu_dir, split="test", token=os.environ.get("HF_TOKEN"))

    items = []
    dropped = 0
    for i, row in enumerate(raw):
        if row["target"] not in TARGET_TO_NUMBER:
            dropped += 1
            continue
        gold = TARGET_TO_NUMBER[row["target"]]
        prompt = PROMPT_TEMPLATES[language].format(
            question=row["question"],
            option1=row["option1"],
            option2=row["option2"],
            option3=row["option3"],
            option4=row["option4"],
        )
        items.append(
            BenchmarkItem(
                benchmark=benchmark_name,
                item_id=f"{benchmark_name}_{i:05d}",
                language=language,
                task_type="classification",
                input={
                    "question": row["question"],
                    "options": [row["option1"], row["option2"], row["option3"], row["option4"]],
                    "domain": row["domain"],
                    "subject": row["subject"],
                },
                gold=gold,
                prompt_template=prompt,
            )
        )

    if dropped:
        print(f"{benchmark_name}: dropped {dropped} item(s) with unrecognized target")

    return items


@register("milu_hi")
def load_milu_hi() -> list[BenchmarkItem]:
    return _load_milu("hi")


@register("milu_bn")
def load_milu_bn() -> list[BenchmarkItem]:
    return _load_milu("bn")