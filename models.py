""" Структуры, классы """

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Record:
    """ Структура для хранения текста у элемента."""
    id: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    """ Структура для сохранения результата поиска.

    id - id записи
    distance - расстояние
    score - сходство
    path - путь к дереву (L/R)
    payload - сам текст
    """
    id: str
    distance: float
    score: float
    path: tuple[str, ...]
    payload: Optional[Record]