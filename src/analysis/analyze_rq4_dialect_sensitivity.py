from pathlib import Path
import pandas as pd


INPUT = Path("results/rq4_pilot_analysis/item_level.csv")
OUT_DIR = Path("results/rq4_pilot_analysis/dialect_sensitivity")

VARIETIES = [
    "standard_hindi",
    "bhojpuri",
    "braj",
    "awadhi",
]


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    required = {
        "model",
        "item_id",
        "instance_id",
        "variety",
        "task",
        "gold",
        "correct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # ------------------------------------------------------------
    # 1. Pivot each semantic item across varieties
    # ------------------------------------------------------------

    pivot = df.pivot_table(
        index=["model", "item_id", "task"],
        columns="variety",
        values="correct",
        aggfunc="first",
    ).reset_index()

    for variety in VARIETIES:
        if variety not in pivot.columns:
            pivot[variety] = pd.NA

    # ------------------------------------------------------------
    # 2. Standard-Hindi baseline vs each dialect
    # ------------------------------------------------------------

    pivot["bhojpuri_drop"] = (
        pivot["standard_hindi"]
        - pivot["bhojpuri"]
    )

    pivot["braj_drop"] = (
        pivot["standard_hindi"]
        - pivot["braj"]
    )

    pivot["awadhi_drop"] = (
        pivot["standard_hindi"]
        - pivot["awadhi"]
    )

    # ------------------------------------------------------------
    # 3. Number of varieties answered correctly
    # ------------------------------------------------------------

    pivot["varieties_correct"] = (
        pivot[VARIETIES]
        .sum(axis=1)
    )

    pivot["all_four_correct"] = (
        pivot["varieties_correct"] == 4
    )

    pivot["standard_only_correct"] = (
        (pivot["standard_hindi"] == 1)
        & (pivot["bhojpuri"] == 0)
        & (pivot["braj"] == 0)
        & (pivot["awadhi"] == 0)
    )

    pivot["any_dialect_failure"] = (
        pivot[
            ["bhojpuri", "braj", "awadhi"]
        ].sum(axis=1) < 3
    )

    # ------------------------------------------------------------
    # 4. Per-model robustness
    # ------------------------------------------------------------

    model_rows = []

    for model, group in pivot.groupby("model"):

        total_items = len(group)

        standard_accuracy = (
            group["standard_hindi"].mean()
        )

        dialect_accuracy = (
            group[
                ["bhojpuri", "braj", "awadhi"]
            ].mean().mean()
        )

        bhojpuri_accuracy = (
            group["bhojpuri"].mean()
        )

        braj_accuracy = (
            group["braj"].mean()
        )

        awadhi_accuracy = (
            group["awadhi"].mean()
        )

        model_rows.append(
            {
                "model": model,
                "items": total_items,
                "standard_hindi_accuracy":
                    standard_accuracy,
                "bhojpuri_accuracy":
                    bhojpuri_accuracy,
                "braj_accuracy":
                    braj_accuracy,
                "awadhi_accuracy":
                    awadhi_accuracy,
                "mean_dialect_accuracy":
                    dialect_accuracy,
                "dialect_gap":
                    standard_accuracy - dialect_accuracy,
                "all_four_correct_items":
                    int(group["all_four_correct"].sum()),
                "items_with_any_dialect_failure":
                    int(group["any_dialect_failure"].sum()),
            }
        )

    model_summary = pd.DataFrame(model_rows)

    for col in [
        "standard_hindi_accuracy",
        "bhojpuri_accuracy",
        "braj_accuracy",
        "awadhi_accuracy",
        "mean_dialect_accuracy",
        "dialect_gap",
    ]:
        model_summary[col] *= 100

    # ------------------------------------------------------------
    # 5. Variety summary
    # ------------------------------------------------------------

    variety_rows = []

    for model, group in pivot.groupby("model"):

        for variety in VARIETIES:

            variety_rows.append(
                {
                    "model": model,
                    "variety": variety,
                    "correct": int(
                        group[variety].sum()
                    ),
                    "total": len(group),
                    "accuracy_percent":
                        group[variety].mean() * 100,
                }
            )

    variety_summary = pd.DataFrame(
        variety_rows
    )

    # ------------------------------------------------------------
    # 6. Task × variety
    # ------------------------------------------------------------

    task_rows = []

    for (
        model,
        task,
    ), group in pivot.groupby(
        ["model", "task"]
    ):

        for variety in VARIETIES:

            task_rows.append(
                {
                    "model": model,
                    "task": task,
                    "variety": variety,
                    "correct": int(
                        group[variety].sum()
                    ),
                    "total": len(group),
                    "accuracy_percent":
                        group[variety].mean() * 100,
                }
            )

    task_summary = pd.DataFrame(
        task_rows
    )

    # ------------------------------------------------------------
    # 7. Item-level dialect transitions
    # ------------------------------------------------------------

    transitions = pivot[
        [
            "model",
            "item_id",
            "task",
            "standard_hindi",
            "bhojpuri",
            "braj",
            "awadhi",
            "bhojpuri_drop",
            "braj_drop",
            "awadhi_drop",
            "varieties_correct",
            "all_four_correct",
            "any_dialect_failure",
        ]
    ].copy()

    # ------------------------------------------------------------
    # 8. Most dialect-sensitive items
    # ------------------------------------------------------------

    transitions["dialect_failures"] = (
        transitions[
            [
                "bhojpuri",
                "braj",
                "awadhi",
            ]
        ].eq(0).sum(axis=1)
    )

    sensitive_items = transitions.sort_values(
        [
            "model",
            "dialect_failures",
        ],
        ascending=[True, False],
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    transitions.to_csv(
        OUT_DIR / "item_level_dialect_sensitivity.csv",
        index=False,
    )

    model_summary.to_csv(
        OUT_DIR / "model_dialect_robustness.csv",
        index=False,
    )

    variety_summary.to_csv(
        OUT_DIR / "variety_summary.csv",
        index=False,
    )

    task_summary.to_csv(
        OUT_DIR / "task_by_variety.csv",
        index=False,
    )

    sensitive_items.to_csv(
        OUT_DIR / "sensitive_items.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------

    print("\n" + "=" * 75)
    print("RQ4 — DIALECT ROBUSTNESS")
    print("=" * 75)

    display = model_summary.copy()

    print(
        display[
            [
                "model",
                "standard_hindi_accuracy",
                "bhojpuri_accuracy",
                "braj_accuracy",
                "awadhi_accuracy",
                "mean_dialect_accuracy",
                "dialect_gap",
                "all_four_correct_items",
                "items_with_any_dialect_failure",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 75)
    print("RQ4 — VARIETY SUMMARY")
    print("=" * 75)

    print(
        variety_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 75)
    print("RQ4 — TASK × VARIETY")
    print("=" * 75)

    print(
        task_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nResults saved to:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()