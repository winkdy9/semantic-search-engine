import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from embedding import Embeddings
from kd_tree import KDTreeIndex
from linear_search import linear_search
from models import Record
from utils import normalize


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataset(data, target_size):
    result = []
    current_id = 1

    while len(result) < target_size:
        item = random.choice(data)
        text = item["text"]

        noisy_text = random.choice([
            text,
            f"  {text}",
            f"{text}   ",
            f"## {text}",
            f"{text} !!!",
            f"...{text}"
        ])

        result.append({
            "id": current_id,
            "text": noisy_text,
        })

        current_id += 1

    return result


def benchmark():
    emb = Embeddings()
    raw_data = load_data("initial_data.json")

    sizes = [100, 500, 1000, 5000, 10000]
    linear_times = []
    tree_times = []

    query_text = "космический аппарат и его траектория"

    print("\nGenerating\n")

    query_vector = normalize(emb.embed(query_text))

    for size in sizes:
        print(f"Benchmark N={size}")

        dataset = prepare_dataset(raw_data, size)
        vectors = []
        payloads = []

        print("Generating")

        for item in dataset:

            vector = normalize(emb.embed(item["text"]))
            vectors.append(vector)

            payloads.append(
                Record(
                    id=item["id"],
                    text=item["text"]
                )
            )

        vectors = np.array(vectors)
        print("Building KD-tree:")

        tree = KDTreeIndex(dim=vectors.shape[1])

        for vector, payload in zip(vectors, payloads):
            tree.insert(
                vector=vector,
                item_id=str(payload.id),
                payload=payload
            )

        print("linear search:")

        start = time.perf_counter()
        linear_search(
            vectors=vectors,
            payloads=payloads,
            query=query_vector,
            k=10
        )

        linear_time = (time.perf_counter() - start) * 1000

        print("Benchmarking KD-tree:")

        start = time.perf_counter()
        tree.search(
            query=query_vector,
            k=10
        )

        tree_time = (time.perf_counter() - start) * 1000

        linear_times.append(linear_time)
        tree_times.append(tree_time)

        speedup = linear_time / tree_time

        print(
            f"N={size} | "
            f"Linear={linear_time:.2f} ms | "
            f"KDTree={tree_time:.2f} ms | "
            f"Speedup={speedup:.2f}x"
        )

        print()

    print("\nFinal Results:\n")

    print(
        f"{'N':<10}"
        f"{'Linear (ms)':<20}"
        f"{'KDTree (ms)':<20}"
        f"{'Speedup':<10}"
    )

    for size, lt, tt in zip(
        sizes,
        linear_times,
        tree_times
    ):

        speedup = lt / tt

        print(
            f"{size:<10}"
            f"{lt:<20.2f}"
            f"{tt:<20.2f}"
            f"{speedup:<10.2f}"
        )

    plt.figure(figsize=(10, 6))

    plt.plot(
        sizes,
        linear_times,
        marker="o",
        label="Linear Search"
    )

    plt.plot(
        sizes,
        tree_times,
        marker="o",
        label="KD-Tree"
    )

    plt.xlabel("N")
    plt.ylabel("Time (ms)")
    plt.title("Linear Search vs KD-Tree")

    plt.legend()

    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    benchmark()