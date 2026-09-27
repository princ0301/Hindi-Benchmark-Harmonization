from pathlib import Path
import pandas as pd


INPUT = Path(
    "results/rq4_pilot_analysis_strict/item_level.csv"
)

OUT_DIR = Path(
    "results/rq4_pilot_analysis_strict/item_audit"
)

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
        "variety",
        "task",
        "gold",
        "correct",
        "raw_output",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # 1. Pivot correctness by variety
    # ---------------------------------------------------------

    correctness = df.pivot_table(
        index=["model", "item_id", "task"],
        columns="variety",
        values="correct",
        aggfunc="first",
    ).reset_index()

    for variety in VARIETIES:
        if variety not in correctness.columns:
            correctness[variety] = pd.NA

    # ---------------------------------------------------------
    # 2. Identify non-uniform items
    # ---------------------------------------------------------

    correctness["pattern"] = (
        correctness[VARIETIES]
        .astype("Int64")
        .astype(str)
        .agg(" | ".join, axis=1)
    )

    correctness["num_correct"] = (
        correctness[VARIETIES]
        .sum(axis=1)
    )

    non_uniform = correctness[
        (correctness["num_correct"] > 0)
        & (correctness["num_correct"] < 4)
    ].copy()

    # ---------------------------------------------------------
    # 3. Classify the pattern
    # ---------------------------------------------------------

    def classify(row):

        s = row["standard_hindi"]
        b = row["bhojpuri"]
        br = row["braj"]
        a = row["awadhi"]

        dialects = [b, br, a]

        # Standard correct, at least one dialect wrong
        if s == 1 and any(x == 0 for x in dialects):
            return "standard_correct_dialect_failure"

        # Standard wrong, at least one dialect correct
        if s == 0 and any(x == 1 for x in dialects):
            return "standard_failure_dialect_success"

        # Mixed dialect pattern while standard remains correct
        if s == 1:
            return "mixed_dialect_pattern"

        # Standard wrong and dialects mixed
        return "mixed_failure_pattern"

    non_uniform["pattern_type"] = (
        non_uniform.apply(
            classify,
            axis=1,
        )
    )

    # ---------------------------------------------------------
    # 4. Identify exact transitions
    # ---------------------------------------------------------

    transition_rows = []

    for _, row in non_uniform.iterrows():

        for variety in [
            "bhojpuri",
            "braj",
            "awadhi",
        ]:

            if (
                row["standard_hindi"] == 1
                and row[variety] == 0
            ):
                transition_rows.append(
                    {
                        "model": row["model"],
                        "item_id": row["item_id"],
                        "task": row["task"],
                        "transition": (
                            "standard_correct_to_dialect_wrong"
                        ),
                        "dialect": variety,
                    }
                )

            elif (
                row["standard_hindi"] == 0
                and row[variety] == 1
            ):
                transition_rows.append(
                    {
                        "model": row["model"],
                        "item_id": row["item_id"],
                        "task": row["task"],
                        "transition": (
                            "standard_wrong_to_dialect_correct"
                        ),
                        "dialect": variety,
                    }
                )

    transitions = pd.DataFrame(
        transition_rows
    )

    # ---------------------------------------------------------
    # 5. Attach actual model outputs
    # ---------------------------------------------------------

    output_table = df[
        [
            "model",
            "item_id",
            "variety",
            "task",
            "gold",
            "correct",
            "raw_output",
        ]
    ].copy()

    audit = non_uniform.merge(
        output_table,
        on=[
            "model",
            "item_id",
            "task",
        ],
        how="left",
        suffixes=("", "_detail"),
    )

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------

    correctness.to_csv(
        OUT_DIR / "all_item_patterns.csv",
        index=False,
    )

    non_uniform.to_csv(
        OUT_DIR / "non_uniform_items.csv",
        index=False,
    )

    transitions.to_csv(
        OUT_DIR / "dialect_transitions.csv",
        index=False,
    )

    audit.to_csv(
        OUT_DIR / "non_uniform_item_outputs.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 7. Print summary
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("RQ4 — ITEM-LEVEL AUDIT")
    print("=" * 75)

    print(
        f"Total model × item groups: "
        f"{len(correctness)}"
    )

    print(
        f"Non-uniform groups: "
        f"{len(non_uniform)}"
    )

    print(
        f"Standard → dialect failures: "
        f"{len(transitions[transitions['transition'] == 'standard_correct_to_dialect_wrong'])}"
    )

    print(
        f"Standard → dialect improvements: "
        f"{len(transitions[transitions['transition'] == 'standard_wrong_to_dialect_correct'])}"
    )

    print("\n" + "=" * 75)
    print("NON-UNIFORM ITEMS")
    print("=" * 75)

    print(
        non_uniform[
            [
                "model",
                "item_id",
                "task",
                "standard_hindi",
                "bhojpuri",
                "braj",
                "awadhi",
                "pattern_type",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("DIALECT TRANSITIONS")
    print("=" * 75)

    if len(transitions) > 0:
        print(
            transitions.to_string(
                index=False
            )
        )
    else:
        print("No transitions found.")

    print("\nResults saved to:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()
