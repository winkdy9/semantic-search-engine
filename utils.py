""" Нормализация и вычисление сходства """
import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """ Нормализация. """
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """ Косинусное сходство. """
    return  1.0 - float(np.dot(a,b))