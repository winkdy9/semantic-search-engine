"""Реализация линейного поиск для сравнения с деревом"""
import heapq

import numpy as np

from models import Record, SearchResult
from utils import cosine_distance, normalize


def linear_search(
    vectors: np.ndarray,
    payloads: list[Record],
    query: np.ndarray,
    k: int
) -> list[SearchResult]:
    """Наивный O(k·N) поиск ближайших соседей.

        vectors - массив векторов,
        payloads - текст,
        query текущий запрос,
        k - параметр
        """

    query = normalize(query)
    heap = []

    for i, vector in enumerate(vectors):
        # Проходка по всем векторам
        vector: np.ndarray
        distance = cosine_distance(query, vector)

        result = SearchResult(
            id=str(payloads[i].id),
            distance=distance,
            score=1.0 - distance,
            path=(),
            payload=payloads[i]
        )

        if len(heap) < k:
            heapq.heappush(
                heap,
                (-distance, i, result)
            )

        else:
            worst_distance = -heap[0][0]

            if distance < worst_distance:
                heapq.heappushpop(
                    heap,
                    (-distance, i, result)
                )

    results = []

    while heap:
        _, _, result = heapq.heappop(heap)
        results.append(result)

    results.reverse()
    return results