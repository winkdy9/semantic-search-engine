import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np

from models import Record, SearchResult
from utils import cosine_distance, normalize


@dataclass
class KDNode:
    """ Структура для узла дерева.

        vector - эмбеддинг
        item_id - айди
        payload - текст
        axis - ось
        left - левые дети
        right - правые дети
    """
    vector: np.ndarray
    item_id: str
    payload: Optional[Record]
    axis: int
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTreeIndex:
    """ Структура дерева """

    def __init__(self, dim: int):
        """ Инициализация.

            dim - размерность
        """
        self.root = None
        self.dim = dim
        self.items = []

    def insert(
            self,
            vector: np.ndarray,
            item_id: str,
            payload: Record | None = None,
    ) -> None:
        """ Добавить вектор в индекс.

            vector - эмбеддинг
            item_id - айди
            payload - текст
            """

        vector = normalize(vector)

        self.items.append((
                vector,
                item_id,
                payload
            ))

    def _build_recursive(
            self,
            items,
            depth
    ):
        """Рекурсивная балансировка дерева"""

        if not items:
            return None

        axis = depth % self.dim
        items.sort(key=lambda x: x[0][axis])

        median = len(items) // 2
        vector, item_id, payload = items[median]

        node = KDNode(
            vector=vector,
            item_id=item_id,
            payload=payload,
            axis=axis
        )

        node.left = self._build_recursive(items[:median],depth + 1)
        node.right = self._build_recursive(items[median + 1:],depth + 1)

        return node

    def build(self):
        """Балансировка дерева"""
        self.root = self._build_recursive(
            self.items,
            depth=0
        )

    def search(
            self,
            query: np.ndarray,
            k: int = 5
    ) -> list[SearchResult]:
        """ Поиск k соседей.

            query - текущая вершина
            k - параметр, количество соседей
            """

        query = normalize(query)
        heap = []

        self._search(
            node=self.root,
            query=query,
            k=k,
            heap=heap,
            path=[]
        )

        results = []

        while heap:
            neg_distance, _, result = heapq.heappop(heap)
            results.append(result)

        results.reverse()

        return results


    def _search(
            self,
            node: Optional[KDNode],
            query: np.ndarray,
            k: int,
            heap: list,
            path: list[str]
    ) -> None:
        """Рекурсивный поиск. Внутренний dfs обход.
            node - вершина
            query текущая точка
            k - параметр
            heap: куча
            path - путь
            """

        if node is None:
            return

        distance = cosine_distance(query, node.vector)

        result = SearchResult(
            id=node.item_id,
            distance=distance,
            score=1.0 - distance,
            path=tuple(path),
            payload=node.payload
        )

        item = (
            -distance,
            node.item_id,
            result
        )

        if len(heap) < k:
            heapq.heappush(heap,item)

        else:
            worst_distance = -heap[0][0]

            if distance < worst_distance:
                heapq.heappushpop(heap, item)

        axis = node.axis
        diff = query[axis] - node.vector[axis]

        if diff < 0:
            near_branch = node.left
            far_branch = node.right

            near_path = path + ['L']
            far_path = path + ['R']

        else:
            near_branch = node.right
            far_branch = node.left

            near_path = path + ['R']
            far_path = path + ['L']

        self._search(
            near_branch,
            query,
            k,
            heap,
            near_path
        )

        worst_distance = float('inf')

        if len(heap) >= k:
            worst_distance = -heap[0][0]

        if abs(diff) < worst_distance:
            self._search(
                far_branch,
                query,
                k,
                heap,
                far_path
            )