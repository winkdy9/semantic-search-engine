import json

from embedding import Embeddings
from kd_tree import KDTreeIndex
from models import Record


def load_data(path: str):
    """Загрузка json"""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    emb = Embeddings()
    data = load_data("initial_data.json")
    first_vector = emb.embed(data[0]["text"])

    index = KDTreeIndex(dim=len(first_vector))

    print("1)   Building index\n")

    for item in data:
        text = item["text"]
        vector = emb.embed(text)

        record = Record(
            id=item["id"],
            text=text
        )

        index.insert(
            vector=vector,
            item_id=str(item["id"]),
            payload=record
        )

    while True:
        query = input("Введите запрос: ")

        if query == "exit":
            break

        query_vector = emb.embed(query)

        results = index.search(
            query=query_vector,
            k=5
        )

        print("\n   Результаты:\n")

        for i, result in enumerate(results, start=1):
            print(
                f"{i}. ",
                f"id={result.id} ",
                f"distance={result.distance:.4f} ",
                f"score={result.score:.4f}"
            )

            if result.payload:
                print(f"   text={result.payload.text}")

            print(f"   path={result.path}")


if __name__ == "__main__":
    main()
