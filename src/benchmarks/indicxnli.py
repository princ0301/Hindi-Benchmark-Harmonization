from src.benchmarks.loader import BenchmarkItem, register

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

PROMPT_TEMPLATES = {
    "hi": (
        "पूर्वधारणा और परिकल्पना के बीच संबंध बताएं। केवल 'entailment', 'neutral', या 'contradiction' लिखें।\n\n"
        "पूर्वधारणा: {premise}\nपरिकल्पना: {hypothesis}\n\nसंबंध:"
    ),
    "bn": (
        "অনুমান এবং প্রকল্পনার মধ্যে সম্পর্ক বলুন। শুধুমাত্র 'entailment', 'neutral', বা 'contradiction' লিখুন।\n\n"
        "অনুমান: {premise}\nপ্রকল্পনা: {hypothesis}\n\nসম্পর্ক:"
    ),
}


def _extract_records(raw) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        list_values = [v for v in raw.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        raise ValueError(f"ambiguous dict structure, list-valued keys: {list(raw.keys())}")
    raise ValueError(f"unexpected top-level type: {type(raw)}")


def _load_indicxnli(language: str) -> list[BenchmarkItem]:
    import json
    from huggingface_hub import hf_hub_download

    benchmark_name = f"indicxnli_{language}"
    local_path = hf_hub_download(
        "Divyanshu/indicxnli", f"forward/test/xnli_{language}.json", repo_type="dataset"
    )
    with open(local_path, encoding="utf-8") as f:
        raw = json.load(f)

    records = _extract_records(raw)

    items = []
    dropped = 0
    for i, row in enumerate(records):
        if row["label"] not in LABEL_MAP:
            dropped += 1
            continue
        gold = LABEL_MAP[row["label"]]
        prompt = PROMPT_TEMPLATES[language].format(premise=row["premise"], hypothesis=row["hypothesis"])
        items.append(
            BenchmarkItem(
                benchmark=benchmark_name,
                item_id=f"{benchmark_name}_{i:04d}",
                language=language,
                task_type="classification",
                input={"premise": row["premise"], "hypothesis": row["hypothesis"]},
                gold=gold,
                prompt_template=prompt,
            )
        )

    if dropped:
        print(f"{benchmark_name}: dropped {dropped} item(s) with unrecognized label")

    return items


@register("indicxnli_hi")
def load_indicxnli_hi() -> list[BenchmarkItem]:
    return _load_indicxnli("hi")


@register("indicxnli_bn")
def load_indicxnli_bn() -> list[BenchmarkItem]:
    return _load_indicxnli("bn")