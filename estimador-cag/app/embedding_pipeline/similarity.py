"""Métricas de similitud entre vectores con la biblioteca estándar (sin numpy).

Para `text-embedding-3-small` los vectores vienen normalizados (norma 1), así que coseno y producto escalar
dan el mismo valor y la distancia euclidiana ordena igual (euclid² = 2 − 2·cos). La regla del curso: usar la
métrica con la que se entrenó el modelo; para OpenAI, coseno o producto escalar son intercambiables.
"""

from __future__ import annotations

import math


def _check_same_dimensions(vec_a: list[float], vec_b: list[float]) -> None:
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vectors must have the same dimensionality ({len(vec_a)} != {len(vec_b)})")
    if not vec_a:
        raise ValueError("Vectors must not be empty")


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Producto escalar. Para vectores normalizados equivale a la similitud coseno."""
    _check_same_dimensions(vec_a, vec_b)
    return sum(a * b for a, b in zip(vec_a, vec_b))


def l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Coseno del ángulo entre los vectores, en [-1, 1]; para embeddings de texto suele caer en [0, 1]."""
    _check_same_dimensions(vec_a, vec_b)
    norm_a, norm_b = l2_norm(vec_a), l2_norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute cosine similarity for a zero-norm vector")
    return dot_product(vec_a, vec_b) / (norm_a * norm_b)


def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Distancia euclidiana: 0 significa idénticos; cuanto menor, más parecidos."""
    _check_same_dimensions(vec_a, vec_b)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))
