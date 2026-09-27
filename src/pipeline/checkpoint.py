import csv
from pathlib import Path


def load_item_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] != "item_id":
                ids.add(row[0])
    return ids


def load_completed_item_ids(scores_path: Path, raw_path: Path) -> set[str]:
    return load_item_ids(scores_path) & load_item_ids(raw_path)