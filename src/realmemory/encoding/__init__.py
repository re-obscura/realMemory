"""Кодирование: эмбеддеры и разреженные SDR (L2)."""
from .embedder import EmbeddingProvider, HashingEmbedder
from .sdr import (
    CalibrationStats,
    SDREncoder,
    calibrate_sparse,
    jaccard,
    overlap,
    overlap_fraction,
)

__all__ = [
    "CalibrationStats",
    "EmbeddingProvider",
    "HashingEmbedder",
    "SDREncoder",
    "calibrate_sparse",
    "jaccard",
    "overlap",
    "overlap_fraction",
]
