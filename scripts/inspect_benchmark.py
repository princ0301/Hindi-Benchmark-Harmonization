import json
import os
import sys

from dotenv import load_dotenv
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()


def inspect_via_script(repo_id: str, config: str | None):
    dataset = load_dataset(repo_id, config)
    print("splits:", list(dataset.keys()))
    split = next(iter(dataset.keys()))
    print("columns:", dataset[split].column_names)
    print("first item:")
    print(dataset[split][0])


def prioritize(files: list[str]) -> list[str]:
    def rank(path: str) -> tuple[int, int]:
        forward_rank = 0 if "forward/" in path and "backward/" not in path else 1
        test_rank = 0 if "/test/" in path else (1 if "/dev/" in path else 2)
        return (forward_rank, test_rank)
    return sorted(files, key=rank)


def load_raw_json(local_path: str):
    with open(local_path, encoding="utf-8") as f:
        content = f.read()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def inspect_via_hub_files(repo_id: str, language_filter: str | None):
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    print("all files in repo:")
    for f in files:
        print(" ", f)

    data_files = [f for f in files if f.endswith((".parquet", ".json", ".jsonl", ".csv", ".tsv"))]
    if language_filter:
        data_files = [
            f for f in data_files
            if f"/{language_filter}." in f or f"-{language_filter}." in f or f"_{language_filter}." in f
        ]
    data_files = prioritize(data_files)

    print("\ndata files matching filter, prioritized:")
    for f in data_files:
        print(" ", f)

    if not data_files:
        return

    target = data_files[0]
    print("\nloading:", target)
    local_path = hf_hub_download(repo_id, target, repo_type="dataset")

    if target.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=local_path)
        split = next(iter(dataset.keys()))
        print("columns:", dataset[split].column_names)
        print("num rows:", len(dataset[split]))
        print("first item:")
        print(dataset[split][0])
        return

    try:
        dataset = load_dataset("json", data_files=local_path)
        split = next(iter(dataset.keys()))
        print("columns:", dataset[split].column_names)
        print("num rows:", len(dataset[split]))
        print("first item:")
        print(dataset[split][0])
    except Exception as e:
        print("datasets JSON loader failed, falling back to raw parse:", e)
        records = load_raw_json(local_path)
        print("type:", type(records))
        if isinstance(records, dict):
            print("top-level keys:", list(records.keys()))
        elif isinstance(records, list):
            print("num records:", len(records))
            print("first record:", records[0])


def inspect_via_data_dir(repo_id: str, data_dir: str, split: str):
    dataset = load_dataset(repo_id, data_dir=data_dir, split=split, token=os.environ.get("HF_TOKEN"))
    print("columns:", dataset.column_names)
    print("num rows:", len(dataset))
    print("first item:")
    print(dataset[0])


def main():
    repo_id = sys.argv[1]
    language_filter = sys.argv[2] if len(sys.argv) > 2 else None

    if len(sys.argv) > 3 and sys.argv[3] == "--data-dir":
        inspect_via_data_dir(repo_id, language_filter, sys.argv[4] if len(sys.argv) > 4 else "test")
        return

    try:
        configs = get_dataset_config_names(repo_id)
        print("available configs:", configs)
        inspect_via_script(repo_id, configs[0] if configs else None)
        return
    except Exception as e:
        print("script-based loading unavailable:", e)

    print("\nfalling back to direct file inspection")
    inspect_via_hub_files(repo_id, language_filter)


if __name__ == "__main__":
    main()