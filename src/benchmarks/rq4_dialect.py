from __future__ import annotations

import csv
from pathlib import Path

from src.benchmarks.loader import BenchmarkItem, register


DATASET_PATH = Path("data/rq4_dialect/rq4_pilot_v0.1_clean.csv")


PROMPT_TEMPLATE = """You are evaluating a Hindi-language benchmark item.

Language variety: {variety}
Task: {task}

Context:
{context}

Question:
{question}

Options:
{options}

Answer the question. For multiple-choice items, return only the option letter (A, B, C, or D). For yes/no items, return only YES or NO. For numeric items, return only the number.

Answer:"""


def _load_csv() -> list[dict[str, str]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"RQ4 dataset not found at {DATASET_PATH}. "
            "Add the cleaned pilot CSV before running the benchmark."
        )

    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "item_id",
        "task",
        "variety",
        "context",
        "question",
        "options",
        "gold_answer",
    }
    if not rows:
        raise ValueError(f"RQ4 dataset is empty: {DATASET_PATH}")

    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"RQ4 dataset is missing columns: {sorted(missing)}")

    return rows


@register("rq4_dialect")
def load_rq4_dialect() -> list[BenchmarkItem]:
    rows = _load_csv()
    items: list[BenchmarkItem] = []

    seen_ids: set[tuple[str, str]] = set()
    for row in rows:
        item_id = row["item_id"].strip()
        variety = row["variety"].strip()
        task = row["task"].strip()

        if not item_id:
            raise ValueError("RQ4 dataset contains an empty item_id")
        
        instance_id = (item_id, variety)
        if instance_id in seen_ids:
            raise ValueError(
                f"Duplicate RQ4 instance: item_id={item_id}, variety={variety}"
            )

        seen_ids.add(instance_id)

        context = row["context"].strip()
        question = row["question"].strip()
        options = row["options"].strip()
        gold = row["gold_answer"].strip()

        prompt = PROMPT_TEMPLATE.format(
            variety=variety,
            task=task,
            context=context or "(none)",
            question=question,
            options=options or "(none)",
        )

        items.append(
            BenchmarkItem(
                benchmark="rq4_dialect",
                item_id=item_id,
                language=variety,
                task_type=task,
                input={
                    "variety": variety,
                    "task": task,
                    "context": context,
                    "question": question,
                    "options": options,
                },
                gold=gold,
                prompt_template=prompt,
            )
        )

    return items