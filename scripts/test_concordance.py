from src.analysis.concordance import build_rank_matrix, kendalls_w, pairwise_kendalls_tau

BENCHMARKS = [
    "indicsentiment_hi",
    "indicxnli_hi",
    "gsm8k_hi",
    "milu_hi",
    "indicquest_hi",
    "ifeval_hi",
    "bfcl_hi",
]

MODEL_SLUGS = [
    "openai_gpt-oss-120b",
    "deepseek_deepseek-chat",
    "google_gemma-2-9b-it",
    "ai4bharat_Airavata",
    "meta-llama_Llama-3.1-8B-Instruct",
]


def main():
    rank_matrix, valid_benchmarks = build_rank_matrix(BENCHMARKS, MODEL_SLUGS)

    print("\nbenchmarks included:", valid_benchmarks)
    print("\nrank matrix (rows=benchmarks, columns=models):")
    print("models:", MODEL_SLUGS)
    for name, row in zip(valid_benchmarks, rank_matrix):
        print(f"  {name}: {row}")

    print("\npairwise Kendall's tau:")
    tau_values = pairwise_kendalls_tau(rank_matrix, valid_benchmarks)
    for (a, b), tau in sorted(tau_values.items(), key=lambda x: x[1]):
        print(f"  {a} vs {b}: tau = {tau:.3f}")

    w, chi2_stat, p_value = kendalls_w(rank_matrix)
    print(f"\nKendall's W = {w:.3f}, chi2 = {chi2_stat:.3f}, p = {p_value:.4f}")


if __name__ == "__main__":
    main()