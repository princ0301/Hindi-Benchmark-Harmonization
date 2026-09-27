from src.benchmarks import indicsentiment
from src.benchmarks.loader import load


def main():
    items = load("indicsentiment_hi")
    print("total items:", len(items))
    print("first item:")
    print(items[0])
    labels = {item.gold for item in items}
    print("distinct labels:", labels)

    none_count = sum(1 for item in items if item.gold is None)
    print("items with None label:", none_count)
    if none_count:
        example = next(item for item in items if item.gold is None)
        print("example None-labeled item:", example)


if __name__ == "__main__":
    main()