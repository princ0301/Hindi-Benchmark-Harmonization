# rq4_dialect_eval.py

import argparse
import json
import re
from pathlib import Path

import pandas as pd


VARIETIES = ["standard_hindi", "bhojpuri", "braj", "awadhi"]


def normalize_text(text):
    """Normalize model output for deterministic matching."""
    if text is None:
        return ""

    text = str(text).strip().lower()

    # Remove common formatting
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = text.strip()

    # Remove common answer prefixes
    text = re.sub(
        r"^(answer|final answer|उत्तर|उत्तर है)\s*[:\-]?\s*",
        "",
        text,
    )

    return text.strip()


def extract_mcq_answer(output):
    """
    Extract A/B/C/D from model output.

    Examples:
        "B" -> B
        "B." -> B
        "The answer is B" -> B
        "(B)" -> B
    """
    text = normalize_text(output)

    match = re.search(r"\b([abcd])\b", text)

    if match:
        return match.group(1).upper()

    # Handle "(b)" / "b." at beginning
    match = re.match(r"^\(?([abcd])\)?[\.\:\)]?", text)

    if match:
        return match.group(1).upper()

    return None


def normalize_special_answer(output, gold):
    """
    Handle deterministic non-MCQ answers such as YES and numbers.
    """
    text = normalize_text(output)
    gold = normalize_text(gold)

    if gold in {"yes", "no"}:
        match = re.search(r"\b(yes|no)\b", text)
        return match.group(1) if match else text

    # Numeric answer
    if re.fullmatch(r"\d+", gold):
        match = re.search(r"\b\d+\b", text)
        return match.group(0) if match else text

    return text


def score_prediction(prediction, gold):
    """
    Score one model prediction.

    Returns:
        1 = correct
        0 = incorrect
    """

    gold_norm = normalize_text(gold)
    pred_norm = normalize_text(prediction)

    # YES / NO
    if gold_norm in {"yes", "no"}:
        pred = normalize_special_answer(prediction, gold)
        return int(pred == gold_norm)

    # Numeric answer
    if re.fullmatch(r"\d+", gold_norm):
        pred = normalize_special_answer(prediction, gold)
        return int(pred == gold_norm)

    # MCQ
    if gold_norm in {"a", "b", "c", "d"}:
        pred = extract_mcq_answer(prediction)
        return int(pred == gold_norm)

    return int(pred_norm == gold_norm)


def load_dataset(path):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

    elif path.suffix.lower() in {".jsonl", ".json"}:
        records = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        df = pd.DataFrame(records)

    else:
        raise ValueError(
            "Dataset must be .csv, .json, or .jsonl"
        )

    required_columns = {
        "item_id",
        "task",
        "variety",
        "context",
        "question",
        "options",
        "gold_answer",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def load_predictions(path):
    """
    Expected prediction CSV:

    item_id,variety,model,output

    Example:

    RC01,bhojpuri,model_1,B
    RC01,bhojpuri,model_2,C
    ...
    """

    df = pd.read_csv(path)

    required = {
        "item_id",
        "variety",
        "model",
        "output",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file missing columns: {sorted(missing)}"
        )

    return df


def evaluate(dataset, predictions):

    merged = predictions.merge(
        dataset[
            [
                "item_id",
                "task",
                "variety",
                "gold_answer",
            ]
        ],
        on=["item_id", "variety"],
        how="left",
        validate="many_to_one",
    )

    if merged["gold_answer"].isna().any():
        bad = merged[merged["gold_answer"].isna()]

        raise ValueError(
            "Some predictions do not match dataset items:\n"
            + str(bad[["item_id", "variety"]])
        )

    merged["correct"] = merged.apply(
        lambda row: score_prediction(
            row["output"],
            row["gold_answer"],
        ),
        axis=1,
    )

    return merged


def make_summary(results):

    # Overall
    overall = (
        results.groupby("model")
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    overall["accuracy"] *= 100

    # By variety
    by_variety = (
        results.groupby(["model", "variety"])
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    by_variety["accuracy"] *= 100

    # By task
    by_task = (
        results.groupby(["model", "task"])
        .agg(
            correct=("correct", "sum"),
            total=("correct", "count"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    by_task["accuracy"] *= 100

    # Model × Variety matrix
    variety_matrix = (
        results.pivot_table(
            index="model",
            columns="variety",
            values="correct",
            aggfunc="mean",
        )
        * 100
    )

    return overall, by_variety, by_task, variety_matrix


def main():

    parser = argparse.ArgumentParser(
        description="RQ4 Regional Hindi Variety Evaluation"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to RQ4 dataset CSV/JSONL",
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to model predictions CSV",
    )

    parser.add_argument(
        "--output_dir",
        default="rq4_results",
        help="Directory for evaluation results",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset)

    print(f"Dataset instances: {len(dataset)}")

    print("\nLoading predictions...")
    predictions = load_predictions(args.predictions)

    print(f"Predictions: {len(predictions)}")

    print("\nEvaluating...")
    results = evaluate(dataset, predictions)

    overall, by_variety, by_task, variety_matrix = make_summary(
        results
    )

    # Save detailed results
    results.to_csv(
        output_dir / "detailed_results.csv",
        index=False,
    )

    overall.to_csv(
        output_dir / "overall_accuracy.csv",
        index=False,
    )

    by_variety.to_csv(
        output_dir / "accuracy_by_variety.csv",
        index=False,
    )

    by_task.to_csv(
        output_dir / "accuracy_by_task.csv",
        index=False,
    )

    variety_matrix.to_csv(
        output_dir / "model_variety_matrix.csv"
    )

    # Print results
    print("\n" + "=" * 70)
    print("RQ4 OVERALL ACCURACY")
    print("=" * 70)

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 70)
    print("ACCURACY BY VARIETY")
    print("=" * 70)

    print(
        by_variety.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 70)
    print("ACCURACY BY TASK")
    print("=" * 70)

    print(
        by_task.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 70)
    print("MODEL × VARIETY")
    print("=" * 70)

    print(
        variety_matrix.to_string(
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nResults saved to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()