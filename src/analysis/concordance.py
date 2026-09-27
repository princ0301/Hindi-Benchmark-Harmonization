import numpy as np
from scipy.stats import chi2, kendalltau

from src.analysis.ranking import load_benchmark_scores, rank_models


def build_rank_matrix(benchmark_names: list[str], model_slugs: list[str]) -> tuple[np.ndarray, list[str]]:
    per_benchmark_ranks = []
    valid_benchmarks = []

    for benchmark_name in benchmark_names:
        scores = load_benchmark_scores(benchmark_name, model_slugs)
        if len(scores) != len(model_slugs):
            missing = set(model_slugs) - set(scores.keys())
            print(f"{benchmark_name}: skipped, missing scores for {missing}")
            continue

        ranks = rank_models(scores)
        per_benchmark_ranks.append([ranks[slug] for slug in model_slugs])
        valid_benchmarks.append(benchmark_name)

    return np.array(per_benchmark_ranks), valid_benchmarks


def pairwise_kendalls_tau(rank_matrix: np.ndarray, benchmark_names: list[str]) -> dict[tuple[str, str], float]:
    tau_values = {}
    m = rank_matrix.shape[0]

    for i in range(m):
        for j in range(i + 1, m):
            tau, _ = kendalltau(rank_matrix[i], rank_matrix[j])
            tau_values[(benchmark_names[i], benchmark_names[j])] = tau

    return tau_values


def kendalls_w(rank_matrix: np.ndarray) -> tuple[float, float, float]:
    m, n = rank_matrix.shape

    rank_sums = rank_matrix.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    s = ((rank_sums - mean_rank_sum) ** 2).sum()

    tie_correction = 0.0
    for row in rank_matrix:
        _, counts = np.unique(row, return_counts=True)
        tie_correction += ((counts**3) - counts).sum()

    denominator = (m**2) * (n**3 - n) - m * tie_correction
    if denominator == 0:
        return 0.0, 0.0, 1.0

    w = (12 * s) / denominator

    chi2_stat = m * (n - 1) * w
    p_value = chi2.sf(chi2_stat, df=n - 1)

    return w, chi2_stat, p_value