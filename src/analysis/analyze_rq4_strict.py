from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.benchmarks.rq4_scorer_v2 import evaluate_rq4

DATASET = Path("data/rq4_dialect/rq4_full_v0.2_120_rows.csv")
RAW_DIR = Path("results/raw_outputs")
OUT_DIR = Path("results/rq4_pilot_analysis_strict")

MODEL_NAMES = {
    "openai_gpt-oss-120b": "GPT-OSS-120B",
    "deepseek_deepseek-chat": "DeepSeek Chat",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "google_gemma-2-9b-it": "Gemma 2 9B IT",
    "ai4bharat_Airavata": "Airavata",
}


def clean_model_name(filename: str) -> str:
    return MODEL_NAMES.get(filename.replace("rq4_dialect__", "").replace(".csv", ""), filename)


def load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATASET)
    required = {"item_id", "variety", "task", "gold_answer"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    dataset["_instance_id"] = dataset["item_id"].astype(str) + "::" + dataset["variety"].astype(str)
    if dataset["_instance_id"].duplicated().any():
        raise ValueError("Duplicate RQ4 instances found")
    return dataset


def analyze_model(raw_path: Path, dataset: pd.DataFrame) -> pd.DataFrame:
    model = clean_model_name(raw_path.name)
    raw = pd.read_csv(raw_path)
    required = {"item_id", "prompt", "output"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{raw_path.name} is missing columns: {sorted(missing)}")

    rows = []
    for _, row in raw.iterrows():
        item_id = str(row["item_id"])
        prompt = str(row["prompt"])
        output = str(row["output"])

        variety = None
        task = None
        for line in prompt.splitlines():
            line = line.strip()
            if line.lower().startswith("language variety:"):
                variety = line.split(":", 1)[1].strip()
            elif line.lower().startswith("task:"):
                task = line.split(":", 1)[1].strip()

        matches = dataset[
            (dataset["item_id"].astype(str) == item_id)
            & (dataset["variety"].astype(str) == variety)
        ]
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely match {item_id}::{variety}")

        gold = str(matches.iloc[0]["gold_answer"])
        evaluation = evaluate_rq4(output, gold)

        rows.append({
            "model": model,
            "item_id": item_id,
            "instance_id": f"{item_id}::{variety}",
            "variety": variety,
            "task": task,
            "gold": gold,
            "predicted": evaluation.predicted,
            "status": evaluation.status,
            "score": evaluation.score,
            "correct": int(evaluation.status == "correct"),
            "raw_output": output,
        })

    return pd.DataFrame(rows)


def summary(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        results.groupby(group_cols)
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            unparseable=("status", lambda x: int((x == "unparseable").sum())),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    out["accuracy_percent"] = out["accuracy"] * 100
    out["unparseable_percent"] = out["unparseable"] / out["total"] * 100
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    raw_files = sorted(RAW_DIR.glob("rq4_dialect__*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No RQ4 raw output files found in {RAW_DIR}")

    results = pd.concat([analyze_model(p, dataset) for p in raw_files], ignore_index=True)

    expected = len(dataset)
    counts = results.groupby("model").size()
    for model, count in counts.items():
        if count != expected:
            raise ValueError(f"{model}: found {count}, expected {expected}")

    overall = summary(results, ["model"])
    by_variety = summary(results, ["model", "variety"])
    by_task = summary(results, ["model", "task"])
    variety_task = summary(results, ["model", "variety", "task"])

    results.to_csv(OUT_DIR / "item_level.csv", index=False)
    overall.to_csv(OUT_DIR / "overall.csv", index=False)
    by_variety.to_csv(OUT_DIR / "by_variety.csv", index=False)
    by_task.to_csv(OUT_DIR / "by_task.csv", index=False)
    variety_task.to_csv(OUT_DIR / "variety_task.csv", index=False)

    print("\n" + "=" * 78)
    print("RQ4 PILOT — STRICT OVERALL")
    print("=" * 78)
    print(overall[["model", "correct", "total", "unparseable", "accuracy_percent", "unparseable_percent"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n" + "=" * 78)
    print("RQ4 PILOT — STRICT BY VARIETY")
    print("=" * 78)
    print(by_variety[["model", "variety", "correct", "total", "unparseable", "accuracy_percent", "unparseable_percent"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n" + "=" * 78)
    print("RQ4 PILOT — STRICT BY TASK")
    print("=" * 78)
    print(by_task[["model", "task", "correct", "total", "unparseable", "accuracy_percent", "unparseable_percent"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nResults saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
