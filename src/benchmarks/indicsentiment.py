from src.benchmarks.loader import BenchmarkItem, register

PROMPT_TEMPLATES = {
    "hi": "निम्नलिखित समीक्षा की भावना बताएं। केवल 'Positive' या 'Negative' लिखें।\n\nसमीक्षा: {review}\n\nभावना:",
    "bn": "নিম্নলিখিত পর্যালোচনার অনুভূতি বলুন। শুধুমাত্র 'Positive' বা 'Negative' লিখুন।\n\nপর্যালোচনা: {review}\n\nঅনুভূতি:",
}


def _load_indicsentiment(language: str) -> list[BenchmarkItem]:
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset

    benchmark_name = f"indicsentiment_{language}"
    local_path = hf_hub_download("ai4bharat/IndicSentiment", f"data/test/{language}.json", repo_type="dataset")
    raw = load_dataset("json", data_files=local_path)["train"]

    items = []
    dropped = 0
    for i, row in enumerate(raw):
        if row["LABEL"] is None:
            dropped += 1
            continue
        prompt = PROMPT_TEMPLATES[language].format(review=row["INDIC REVIEW"])
        items.append(
            BenchmarkItem(
                benchmark=benchmark_name,
                item_id=f"{benchmark_name}_{i:04d}",
                language=language,
                task_type="classification",
                input={"review": row["INDIC REVIEW"]},
                gold=row["LABEL"],
                prompt_template=prompt,
            )
        )

    if dropped:
        print(f"{benchmark_name}: dropped {dropped} item(s) with missing label")

    return items


@register("indicsentiment_hi")
def load_indicsentiment_hi() -> list[BenchmarkItem]:
    return _load_indicsentiment("hi")


@register("indicsentiment_bn")
def load_indicsentiment_bn() -> list[BenchmarkItem]:
    return _load_indicsentiment("bn")