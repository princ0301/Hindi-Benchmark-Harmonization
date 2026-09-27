from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BenchmarkItem:
    benchmark: str
    item_id: str
    language: str
    task_type: str
    input: dict[str, Any]
    gold: Any
    prompt_template: str


LOADERS: dict[str, Callable[[], list[BenchmarkItem]]] = {}


def register(name: str):
    def wrapper(fn: Callable[[], list[BenchmarkItem]]):
        LOADERS[name] = fn
        return fn
    return wrapper


def load(benchmark_name: str) -> list[BenchmarkItem]:
    if benchmark_name not in LOADERS:
        raise ValueError(f"no loader registered for '{benchmark_name}'")
    return LOADERS[benchmark_name]()