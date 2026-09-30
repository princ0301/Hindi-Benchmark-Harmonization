# Hindi Benchmark Harmonization

A comprehensive evaluation framework for assessing Hindi language LLM performance across multiple benchmarks and regional varieties. This project conducts cross-benchmark consistency studies to harmonize and understand how different LLMs perform on Hindi NLP tasks.

## Overview

This repository implements a unified evaluation pipeline that tests various Large Language Models (LLMs) against standardized Hindi NLP benchmarks including sentiment analysis, question answering, entailment recognition, and regional dialect understanding. The framework supports both API-based and local model inference with automatic scoring and result aggregation.

## Stack

- **Language:** Python 3.12
- **Framework:** Modular pipeline architecture with plugin-based benchmark loaders
- **Key Libraries:**
  - `datasets` — HuggingFace dataset loading and management
  - `tenacity` — Automatic retry logic with exponential backoff
  - `pandas` — Result aggregation and analysis

## Repository Structure

```
.
├── configs/                 Configuration files (benchmarks, models, run settings)
├── data/                    Raw benchmark data
├── evaluation/              Dialect evaluation and result analysis scripts
├── results/                 Generated outputs (raw model outputs and scores)
├── scripts/                 Utility and testing scripts
│   ├── inspect_benchmark.py  Dataset introspection tool
│   ├── smoke_test_*.py       Quick validation scripts for models and benchmarks
│   ├── test_concordance*.py  Cross-benchmark consistency tests
│   └── *_api.py              Individual model provider test scripts
├── src/
│   ├── benchmarks/          Benchmark implementations and scorers
│   │   ├── loader.py        Benchmark registration and loading system
│   │   ├── indicsentiment.py Sentiment classification (Hindi & Bengali)
│   │   ├── indicxnli.py     Natural language inference
│   │   ├── indicquest.py    Question answering
│   │   ├── gsm8k.py         Math reasoning
│   │   ├── bfcl.py          Function calling
│   │   ├── ifeval.py        Instruction following
│   │   ├── chatrag.py       RAG conversations
│   │   ├── milu.py          Multilingual tasks
│   │   ├── rq4_dialect.py   Regional Hindi varieties
│   │   └── scorers.py       Unified scoring system (exact match, BLEU, LLM-judge)
│   ├── models/              LLM client abstractions
│   │   ├── base.py          Abstract ModelClient interface
│   │   ├── api_clients.py   API-based clients (Gemini, Groq, OpenRouter, NVIDIA NIM, Together, Ollama)
│   │   └── local_clients.py Local GPU inference wrapper
│   └── pipeline/            Evaluation orchestration
│       ├── run_eval.py      Main evaluation entrypoint
│       ├── checkpoint.py    Resume-on-failure checkpointing
│       ├── rate_limiter.py  API rate limiting
│       └── remote_gpu.py    Remote GPU orchestration
├── tests/                   Test suite
│   ├── test_scorers.py      Scoring function validation
│   ├── test_rq4_scorer_v2.py Regional dialect scoring tests
│   └── test_together_api.py  API integration tests
├── pyproject.toml           Project metadata and dependencies
├── uv.lock                  Locked dependency versions
└── .python-version          Python 3.12 specification
```

## Key Features

### Supported Benchmarks

| Benchmark | Task Type | Language | Size |
|-----------|-----------|----------|------|
| **IndicSentiment** | Classification | Hindi, Bengali | 200 samples |
| **IndicXNLI** | Natural Language Inference | Hindi, Bengali | 200 samples |
| **IndicQuest** | Question Answering | Hindi | 200 samples |
| **GSM8K** | Math Reasoning | Hindi | 200 samples |
| **MILU** | Multilingual Understanding | Hindi, Bengali | 200 samples |
| **BFCL** | Function Calling | Hindi | 999 samples |
| **IFEval** | Instruction Following | Hindi | 200 samples |
| **ChatRAG** | RAG Conversations | Hindi | - |
| **RQ4 Dialect** | Regional Varieties | Hindi (Standard, Bhojpuri, Braj, Awadhi) | 40 samples each |

### Scoring Methods

- **Exact Match** — Character-level comparison with normalization
- **BLEU Score** — N-gram overlap metric for generation tasks
- **LLM Judge** — Using Gemma4:31B model for semantic evaluation
- **MCQ Extraction** — Robust A/B/C/D answer extraction with fuzzy matching
- **YES/NO Detection** — Binary answer normalization
- **Numeric Matching** — Integer answer comparison

### Regional Dialect Support

RQ4 Dialect evaluation covers:
- **Standard Hindi** — Modern formal Hindi
- **Bhojpuri** — Eastern Hindi variety (Bihar/Uttar Pradesh)
- **Braj** — Central Hindi variety (Mathura region)
- **Awadhi** — Eastern Hindi variety (Awadh region)

## How to Run

### Setup

```bash
# Clone repository
git clone https://github.com/princ0301/Hindi-Benchmark-Harmonization.git
cd Hindi-Benchmark-Harmonization

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Run Benchmark Evaluation

```bash
# Evaluate a single benchmark
python -m src.pipeline.run_eval indicsentiment_hi

# With custom sample size
python -m src.pipeline.run_eval indicsentiment_hi 100

# Available benchmarks
python -m src.pipeline.run_eval indicxnli_hi
python -m src.pipeline.run_eval indicquest_hi
python -m src.pipeline.run_eval gsm8k_hi
python -m src.pipeline.run_eval bfcl_hi
python -m src.pipeline.run_eval ifeval_hi
python -m src.pipeline.run_eval milu_hi
python -m src.pipeline.run_eval rq4_dialect
python -m src.pipeline.run_eval indicsentiment_bn
python -m src.pipeline.run_eval indicxnli_bn
python -m src.pipeline.run_eval milu_bn
```

### Evaluate Regional Dialects (RQ4)

```bash
# Run dialect evaluation with predictions
python evaluation/rq4_dialect_eval.py \
  --dataset data/rq4_dataset.jsonl \
  --predictions results/rq4_predictions.csv \
  --output_dir results/rq4_results
```

Expected prediction CSV format:
```csv
item_id,variety,model,output
RC01,bhojpuri,model_1,B
RC01,standard_hindi,model_1,A
```

### Inspect Datasets

```bash
# List dataset contents
python scripts/inspect_benchmark.py ai4bharat/IndicSentiment hi

# With specific data directory
python scripts/inspect_benchmark.py ai4bharat/IndicSentiment hi --data-dir test
```

### Run Tests

```bash
# All tests
python -m pytest tests/

# Specific test
python -m pytest tests/test_scorers.py -v

# Test scoring functions
python -m pytest tests/test_rq4_scorer_v2.py::test_mcq_extraction -v
```

### Smoke Tests

```bash
# Test model connectivity
python scripts/smoke_test_models.py

# Test sentiment benchmark
python scripts/smoke_test_indicsentiment.py

# Test API providers
python scripts/together_api.py
python scripts/nim.py
python scripts/cerebral.py
python scripts/cloudflare_generate.py
```

## Results Structure

Evaluation outputs organized by benchmark × model:

```
results/
├── raw_outputs/
│   ├── indicsentiment_hi__groq_*.csv    (item_id, prompt, output)
│   ├── indicsentiment_hi__openrouter_*.csv
│   └── ...
└── scores/
    ├── indicsentiment_hi__groq_*.csv    (item_id, gold, score)
    ├── indicsentiment_hi__openrouter_*.csv
    └── ...
```

Resume on failure: Script automatically loads completed items from score files and skips them on restart.

## RQ4 Dialect Evaluation Output

```
rq4_results/
├── detailed_results.csv        All predictions with correctness
├── overall_accuracy.csv        Model × Accuracy summary
├── accuracy_by_variety.csv     Model × Variety × Accuracy
├── accuracy_by_task.csv        Model × Task × Accuracy
└── model_variety_matrix.csv    Heatmap-ready results
```

## Benchmark Registration System

Add new benchmarks via the decorator pattern:

```python
from src.benchmarks.loader import BenchmarkItem, register

@register("my_benchmark_hi")
def load_my_benchmark() -> list[BenchmarkItem]:
    items = []
    for row in load_data():
        items.append(BenchmarkItem(
            benchmark="my_benchmark_hi",
            item_id=f"my_benchmark_hi_{i:04d}",
            language="hi",
            task_type="classification",
            input={"text": row["input"]},
            gold=row["label"],
            prompt_template=f"Classify: {row['input']}"
        ))
    return items
```

## Scoring Customization

Extend scoring in `src/benchmarks/scorers.py`:

```python
def score_my_task(output: str, gold: Any) -> float:
    """Score output against gold standard. Return [0, 1]."""
    return 1.0 if output.strip().lower() == str(gold).lower() else 0.0

# Register in score_item() dispatcher
if benchmark_name == "my_benchmark_hi":
    return score_my_task(output, item.gold)
```

## Project Structure Notes

- **Stateless benchmarks:** Each benchmark loader is self-contained and downloads data on first run
- **Checkpointing:** `src/pipeline/checkpoint.py` tracks completed items to enable resume-on-failure
- **Rate limiting:** `src/pipeline/rate_limiter.py` prevents API throttling for rate-limited providers like OpenRouter
- **Concurrent evaluation:** ThreadPoolExecutor runs multiple models in parallel with shared result directories
- **Error recovery:** Retries with exponential backoff via `tenacity`; individual item errors don't halt the pipeline

## Try Asking

- "How do I add support for a new Hindi benchmark dataset?"
- "What's the accuracy difference between Gemma and Llama on IndicSentiment across regional dialects?"
- "How do I run offline evaluation with locally hosted models via Ollama?"

## Support

For issues or questions:
- Open a GitHub issue
- Check existing benchmark documentation in `src/benchmarks/`
- Review test cases in `tests/` for usage examples
