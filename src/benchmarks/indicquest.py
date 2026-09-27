import csv
import io

from src.benchmarks.loader import BenchmarkItem, register

INDICQUEST_URL = "https://raw.githubusercontent.com/l3cube-pune/indic-nlp/main/L3Cube-IndicQuest/Hindi.csv"

QA_PROMPT_TEMPLATE = "निम्नलिखित प्रश्न का उत्तर हिंदी में संक्षेप में दें।\n\nप्रश्न: {question}\n\nउत्तर:"


@register("indicquest_hi")
def load_indicquest_hi() -> list[BenchmarkItem]:
    import urllib.request

    with urllib.request.urlopen(INDICQUEST_URL) as response:
        raw_text = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(raw_text))

    items = []
    dropped = 0
    for i, row in enumerate(reader):
        question = row.get("Question", "").strip()
        answer = row.get("Answer", "").strip()
        if not question or not answer:
            dropped += 1
            continue
        prompt = QA_PROMPT_TEMPLATE.format(question=question)
        items.append(
            BenchmarkItem(
                benchmark="indicquest_hi",
                item_id=f"indicquest_hi_{i:04d}",
                language="hi",
                task_type="generation",
                input={"question": question, "domain": row.get("Domain", "").strip()},
                gold=answer,
                prompt_template=prompt,
            )
        )

    if dropped:
        print(f"indicquest_hi: dropped {dropped} item(s) with missing question or answer")

    return items