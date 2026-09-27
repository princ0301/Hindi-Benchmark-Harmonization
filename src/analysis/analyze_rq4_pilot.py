from pathlib import Path
import pandas as pd

from src.benchmarks.scorers import score_rq4_dialect


DATASET = Path("data/rq4_dialect/rq4_pilot_v0.1_clean.csv")
RAW_DIR = Path("results/raw_outputs")
OUT_DIR = Path("results/rq4_pilot_analysis")


MODEL_NAMES = {
    "openai_gpt-oss-120b": "GPT-OSS-120B",
    "deepseek_deepseek-chat": "DeepSeek Chat",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "google_gemma-2-9b-it": "Gemma 2 9B IT",
    "ai4bharat_Airavata": "Airavata",
}


def clean_model_name(filename: str) -> str:
    name = (
        filename
        .replace("rq4_dialect__", "")
        .replace(".csv", "")
    )

    return MODEL_NAMES.get(name, name)


def load_dataset():
    dataset = pd.read_csv(DATASET)

    required = {
        "item_id",
        "variety",
        "task",
        "gold_answer",
    }

    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    # True RQ4 instance identity is item_id + variety.
    dataset["_instance_id"] = (
        dataset["item_id"].astype(str)
        + "::"
        + dataset["variety"].astype(str)
    )

    if dataset["_instance_id"].duplicated().any():
        duplicates = dataset.loc[
            dataset["_instance_id"].duplicated(keep=False),
            ["item_id", "variety"],
        ]

        raise ValueError(
            "Duplicate RQ4 instances found:\n"
            + duplicates.to_string(index=False)
        )

    return dataset


def analyze_model(raw_path: Path, dataset: pd.DataFrame):

    model = clean_model_name(raw_path.name)

    raw = pd.read_csv(raw_path)

    required = {
        "item_id",
        "prompt",
        "output",
    }

    missing = required - set(raw.columns)

    if missing:
        raise ValueError(
            f"{raw_path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    rows = []

    for _, row in raw.iterrows():

        item_id = str(row["item_id"])
        prompt = str(row["prompt"])
        output = str(row["output"])

        # The RQ4 prompt contains the variety and task.
        variety = None
        task = None

        for line in prompt.splitlines():

            line = line.strip()

            if line.lower().startswith("language variety:"):
                variety = line.split(":", 1)[1].strip()

            elif line.lower().startswith("task:"):
                task = line.split(":", 1)[1].strip()

        if variety is None or task is None:
            raise ValueError(
                f"Could not recover variety/task from prompt "
                f"for {model} / {item_id}"
            )

        matches = dataset[
            (dataset["item_id"].astype(str) == item_id)
            & (dataset["variety"].astype(str) == variety)
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Could not uniquely match dataset instance: "
                f"item_id={item_id}, variety={variety}, "
                f"matches={len(matches)}"
            )

        dataset_row = matches.iloc[0]

        gold = str(dataset_row["gold_answer"])

        # IMPORTANT:
        # Use EXACTLY the same scorer used by run_eval.py.
        score = score_rq4_dialect(
            output,
            gold,
        )

        correct = int(score == 1.0)

        instance_id = (
            f"{item_id}::{variety}"
        )

        rows.append(
            {
                "model": model,
                "item_id": item_id,
                "instance_id": instance_id,
                "variety": variety,
                "task": task,
                "gold": gold,
                "score": score,
                "correct": correct,
                "raw_output": output,
            }
        )

    return pd.DataFrame(rows)


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset()

    raw_files = sorted(
        RAW_DIR.glob("rq4_dialect__*.csv")
    )

    if not raw_files:
        raise FileNotFoundError(
            f"No RQ4 raw output files found in {RAW_DIR}"
        )

    all_results = []

    for raw_path in raw_files:

        print(
            f"Analyzing {clean_model_name(raw_path.name)}..."
        )

        model_results = analyze_model(
            raw_path,
            dataset,
        )

        all_results.append(model_results)

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Verify expected evaluation count
    # ---------------------------------------------------------

    expected_per_model = len(dataset)

    counts = (
        results
        .groupby("model")
        .size()
    )

    for model, count in counts.items():

        if count != expected_per_model:
            raise ValueError(
                f"{model}: found {count} evaluations, "
                f"expected {expected_per_model}"
            )

    # ---------------------------------------------------------
    # Overall
    # ---------------------------------------------------------

    overall = (
        results
        .groupby("model")
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    overall["accuracy_percent"] = (
        overall["accuracy"] * 100
    )

    # ---------------------------------------------------------
    # By variety
    # ---------------------------------------------------------

    by_variety = (
        results
        .groupby(
            ["model", "variety"]
        )
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    by_variety["accuracy_percent"] = (
        by_variety["accuracy"] * 100
    )

    # ---------------------------------------------------------
    # By task
    # ---------------------------------------------------------

    by_task = (
        results
        .groupby(
            ["model", "task"]
        )
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    by_task["accuracy_percent"] = (
        by_task["accuracy"] * 100
    )

    # ---------------------------------------------------------
    # Variety × Task
    # ---------------------------------------------------------

    variety_task = (
        results
        .groupby(
            ["model", "variety", "task"]
        )
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    variety_task["accuracy_percent"] = (
        variety_task["accuracy"] * 100
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    results.to_csv(
        OUT_DIR / "item_level.csv",
        index=False,
    )

    overall.to_csv(
        OUT_DIR / "overall.csv",
        index=False,
    )

    by_variety.to_csv(
        OUT_DIR / "by_variety.csv",
        index=False,
    )

    by_task.to_csv(
        OUT_DIR / "by_task.csv",
        index=False,
    )

    variety_task.to_csv(
        OUT_DIR / "variety_task.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RQ4 PILOT — OVERALL")
    print("=" * 70)

    print(
        overall[
            [
                "model",
                "correct",
                "total",
                "accuracy_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 70)
    print("RQ4 PILOT — BY VARIETY")
    print("=" * 70)

    print(
        by_variety[
            [
                "model",
                "variety",
                "correct",
                "total",
                "accuracy_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 70)
    print("RQ4 PILOT — BY TASK")
    print("=" * 70)

    print(
        by_task[
            [
                "model",
                "task",
                "correct",
                "total",
                "accuracy_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nResults saved to:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()