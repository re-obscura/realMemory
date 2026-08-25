"""Кодирование: эмбеддеры, биполярные адреса (L1), разреженные SDR (L2)."""
from .embedder import EmbeddingProvider, HashingEmbedder
from .sdr import (
    BipolarProjector,
    CalibrationStats,
    SDREncoder,
    calibrate_sparse,
    jaccard,
    overlap,
    overlap_fraction,
)

__all__ = [
    "BipolarProjector",
    "CalibrationStats",
    "EmbeddingProvider",
    "HashingEmbedder",
    "SDREncoder",
    "calibrate_sparse",
    "jaccard",
    "overlap",
    "overlap_fraction",
]
