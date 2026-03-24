"""Faithfulness probes for attribution quality."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _safe_float(value: float) -> float:
    """Clamp a numeric value to [0, 1].

    Args:
        value: Numeric value.

    Returns:
        Clamped float between 0 and 1.

    Raises:
        None.
    """
    return float(max(0.0, min(1.0, value)))


def _model_confidence(model, token_ids: Sequence[int]) -> float:
    """Query model confidence from a flexible model interface.

    Args:
        model: Model object or callable.
        token_ids: Token id sequence.

    Returns:
        Confidence score in [0, 1] after clamping.

    Raises:
        AttributeError: If model lacks supported prediction interface.
    """
    if callable(model):
        return _safe_float(float(model(token_ids)))
    if hasattr(model, "tool_confidence"):
        return _safe_float(float(model.tool_confidence(token_ids)))
    if hasattr(model, "predict_confidence"):
        return _safe_float(float(model.predict_confidence(token_ids)))
    raise AttributeError("model must be callable or provide tool_confidence/predict_confidence")


def sufficiency(model, token_ids: Sequence[int], top_k_mask: Sequence[bool]) -> float:
    """Measure confidence retained by only top-k attributed tokens.

    Args:
        model: Confidence model interface.
        token_ids: Full token id sequence.
        top_k_mask: Boolean mask indicating top-k tokens to keep.

    Returns:
        Sufficiency score in [0, 1].

    Raises:
        ValueError: If mask length differs from token length.
    """
    if len(token_ids) != len(top_k_mask):
        raise ValueError("token_ids and top_k_mask must have equal length")

    kept = [tok for tok, keep in zip(token_ids, top_k_mask) if keep]
    if not kept:
        return 0.0
    return _model_confidence(model, kept)


def comprehensiveness(model, token_ids: Sequence[int], top_k_mask: Sequence[bool]) -> float:
    """Measure confidence drop after removing top-k attributed tokens.

    Args:
        model: Confidence model interface.
        token_ids: Full token id sequence.
        top_k_mask: Boolean mask for top-k tokens to remove.

    Returns:
        Comprehensiveness score in [0, 1].

    Raises:
        ValueError: If mask length differs from token length.
    """
    if len(token_ids) != len(top_k_mask):
        raise ValueError("token_ids and top_k_mask must have equal length")

    full = _model_confidence(model, token_ids)
    reduced = [tok for tok, keep in zip(token_ids, top_k_mask) if not keep]
    if not reduced:
        reduced_conf = 0.0
    else:
        reduced_conf = _model_confidence(model, reduced)

    return _safe_float(full - reduced_conf)
