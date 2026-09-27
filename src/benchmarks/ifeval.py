from src.benchmarks.loader import BenchmarkItem, register


@register("ifeval_hi")
def load_ifeval_hi() -> list[BenchmarkItem]:
    from datasets import load_dataset

    raw = load_dataset("nvidia/IFEval-Hi", split="train")

    items = []
    for row in raw:
        items.append(
            BenchmarkItem(
                benchmark="ifeval_hi",
                item_id=f"ifeval_hi_{row['key']:05d}",
                language="hi",
                task_type="generation",
                input={
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                },
                gold=None,
                prompt_template=row["prompt"],
            )
        )

    return items