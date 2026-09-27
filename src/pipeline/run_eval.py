import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from src.benchmarks import bfcl, chatrag, gsm8k, ifeval, indicquest, indicsentiment, indicxnli, milu
from src.benchmarks.loader import load
from src.benchmarks.scorers import score_bfcl, score_ifeval, score_item, score_llm_judge
from src.models.api_clients import GeminiClient, GroqClient, NvidiaNimClient, OllamaClient, OpenRouterClient, TogetherClient
from src.models.local_clients import RemoteGPUClient
from src.pipeline.checkpoint import load_completed_item_ids

load_dotenv()

RESULTS_DIR = Path("results")

JUDGE_BASED_BENCHMARKS = {"indicquest_hi"}
RULE_BASED_BENCHMARKS = {"ifeval_hi"}
FUNCTION_CALL_BENCHMARKS = {"bfcl_hi"}

SAMPLE_SEED = 42
DEFAULT_SAMPLE_SIZE = 200
SAMPLE_SIZES = {
    "indicsentiment_hi": 200,
    "indicxnli_hi": 200,
    "gsm8k_hi": 200,
    "milu_hi": 200,
    "indicquest_hi": 200,
    "bfcl_hi": 999999,   # run full dataset; capped automatically to actual size below
    "ifeval_hi": 200,
    # "chatrag_hi": 200,  # excluded: cost/rate-limit constraints, see paper Limitations
    "indicsentiment_bn": 200,
    "indicxnli_bn": 200,
    "milu_bn": 200,
}


def get_judge_client():
    return OllamaClient()


def get_models():
    return [
        # GeminiClient(),  # paused: hit free-tier quota limit, resume later
        GroqClient(),
        OpenRouterClient(model_name="deepseek/deepseek-chat"),
        RemoteGPUClient(hf_model_id="meta-llama/Llama-3.1-8B-Instruct"),
        RemoteGPUClient(hf_model_id="google/gemma-2-9b-it", load_in_4bit=True),
        RemoteGPUClient(
            hf_model_id="ai4bharat/Airavata",
            prompt_format="<|user|>\n{prompt}\n<|assistant|>\n",
        ),
    ]


GENERATION_SETTINGS = {
    "gsm8k_hi": {"max_new_tokens": 1024},
    "indicquest_hi": {"max_new_tokens": 1024},
    "ifeval_hi": {"max_new_tokens": 2048},
    "chatrag_hi": {"max_new_tokens": 1024},
}
DEFAULT_MAX_NEW_TOKENS = 512


def run_model(benchmark_name: str, model, items) -> float:
    raw_dir = RESULTS_DIR / "raw_outputs"
    scores_dir = RESULTS_DIR / "scores"
    raw_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = GENERATION_SETTINGS.get(benchmark_name, {}).get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

    model_slug = model.model_id.replace("/", "_")
    raw_path = raw_dir / f"{benchmark_name}__{model_slug}.csv"
    scores_path = scores_dir / f"{benchmark_name}__{model_slug}.csv"

    completed = load_completed_item_ids(scores_path, raw_path)
    remaining = [item for item in items if item.item_id not in completed]

    if completed:
        print(f"{model.model_id}: resuming, {len(completed)} item(s) already done, {len(remaining)} remaining")

    raw_needs_header = not raw_path.exists() or raw_path.stat().st_size == 0
    scores_needs_header = not scores_path.exists() or scores_path.stat().st_size == 0

    judge_client = get_judge_client() if benchmark_name in JUDGE_BASED_BENCHMARKS else None

    with open(raw_path, "a", newline="", encoding="utf-8") as raw_f, \
         open(scores_path, "a", newline="", encoding="utf-8") as scores_f:

        raw_writer = csv.writer(raw_f)
        scores_writer = csv.writer(scores_f)

        if raw_needs_header:
            raw_writer.writerow(["item_id", "prompt", "output"])
        if scores_needs_header:
            scores_writer.writerow(["item_id", "gold", "score"])

        for idx, item in enumerate(remaining, start=1):
            try:
                output = model.generate(item.prompt_template, max_new_tokens=max_new_tokens)

                if judge_client is not None:
                    score = score_llm_judge(item.input["question"], item.gold, output, judge_client)
                elif benchmark_name in RULE_BASED_BENCHMARKS:
                    score = score_ifeval(output, item.input["instruction_id_list"], item.input["kwargs"])
                elif benchmark_name in FUNCTION_CALL_BENCHMARKS:
                    score = score_bfcl(output, item.input["function_schema"], item.input["ground_truth"])
                else:
                    score = score_item(benchmark_name, output, item.gold)
            except Exception as e:
                print(f"{model.model_id}: SKIPPING item {item.item_id} after unrecoverable error: {e}")
                continue

            if item.gold is not None and isinstance(item.gold, list):
                gold_field = json.dumps(item.gold, ensure_ascii=False)
            elif item.gold is not None:
                gold_field = item.gold
            elif "instruction_id_list" in item.input:
                gold_field = ";".join(item.input["instruction_id_list"])
            elif "ground_truth" in item.input:
                gold_field = json.dumps(item.input["ground_truth"], ensure_ascii=False)
            else:
                gold_field = ""

            raw_writer.writerow([item.item_id, item.prompt_template, output])
            scores_writer.writerow([item.item_id, gold_field, score])
            raw_f.flush()
            scores_f.flush()

            if idx % 5 == 0 or idx == len(remaining):
                print(f"{model.model_id}: {idx}/{len(remaining)} items done")

    scored_rows = [
        float(row[2]) for row in csv.reader(open(scores_path, encoding="utf-8"))
        if row and row[0] != "item_id"
    ]

    if len(scored_rows) != len(items):
        print(
            f"{model.model_id}: NOTE - scores file has {len(scored_rows)} rows "
            f"but {len(items)} items were requested. This can mean duplicates/a partial "
            f"write (investigate if unexpected), OR one or more items were permanently "
            f"skipped due to an unrecoverable per-item error (e.g. prompt too large for "
            f"a model's context/rate-limit tier) — check the SKIPPING log lines above."
        )

    accuracy = sum(scored_rows) / len(scored_rows) if scored_rows else 0.0

    if not 0.0 <= accuracy <= 1.0:
        print(f"{model.model_id}: WARNING - accuracy {accuracy:.3f} is out of the valid [0, 1] range")

    return accuracy, len(scored_rows)


def run(benchmark_name: str, sample_size: int | None = None):
    all_items = load(benchmark_name)

    target_size = sample_size if sample_size is not None else SAMPLE_SIZES.get(benchmark_name, DEFAULT_SAMPLE_SIZE)
    target_size = min(target_size, len(all_items))

    rng = random.Random(SAMPLE_SEED)
    items = rng.sample(all_items, target_size)

    models = get_models()

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(run_model, benchmark_name, model, items): model
            for model in models
        }
        for future in as_completed(futures):
            model = futures[future]
            accuracy, actual_row_count = future.result()
            print(f"{model.model_id}: accuracy = {accuracy:.3f} on {actual_row_count} items (target was {len(items)})")


if __name__ == "__main__":
    benchmark_name = sys.argv[1] if len(sys.argv) > 1 else "indicsentiment_hi"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(benchmark_name, sample_size)