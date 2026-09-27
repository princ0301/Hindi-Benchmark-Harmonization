"""Run RQ4 using the strict parser without changing other benchmark scorers."""

from __future__ import annotations

import sys

from src.benchmarks import scorers
from src.benchmarks.rq4_scorer_v2 import evaluate_rq4
from src.pipeline.run_eval import run


def strict_score(model_output: str, gold: str) -> float:
    return evaluate_rq4(model_output, gold).score


scorers.SCORERS["rq4_dialect"] = strict_score


if __name__ == "__main__":
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run("rq4_dialect", sample_size)
