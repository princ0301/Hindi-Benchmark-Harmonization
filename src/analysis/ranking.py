import csv
from pathlib import Path

from scipy.stats import rankdata

RESULTS_DIR = Path("results")


def load_mean_score(benchmark_name: str, model_slug: str) -> float | None:
    path = RESULTS_DIR / "scores" / f"{benchmark_name}__{model_slug}.csv"
    if not path.exists():
        return None

    scores = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] != "item_id":
                scores.append(float(row[2]))

    return sum(scores) / len(scores) if scores else None


def load_benchmark_scores(benchmark_name: str, model_slugs: list[str]) -> dict[str, float]:
    scores = {}
    for slug in model_slugs:
        mean_score = load_mean_score(benchmark_name, slug)
        if mean_score is not None:
            scores[slug] = mean_score
    return scores


def rank_models(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    model_slugs = list(scores.keys())
    values = [scores[slug] for slug in model_slugs]

    ranks = rankdata([-v for v in values], method="average")

    return dict(zip(model_slugs, ranks))