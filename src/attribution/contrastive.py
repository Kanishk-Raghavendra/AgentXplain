"""Contrastive attribution utilities for tool selection."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np


def contrastive_attribution(
    attribution_fn: Callable[[str], np.ndarray],
    tokens: Sequence[str],
    selected_tool: str,
    alternative_tool: str,
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Compute contrastive token saliency between selected and alternative tools.

    Args:
        attribution_fn: Function mapping a tool name to token saliency array.
        tokens: Token sequence aligned with returned saliency arrays.
        selected_tool: Chosen tool label.
        alternative_tool: Rival tool label used for contrast.

    Returns:
        Tuple of contrastive saliency and ranked token-score list.

    Raises:
        ValueError: If returned array lengths do not match tokens.
    """
    selected_scores = np.asarray(attribution_fn(selected_tool), dtype=np.float32)
    alternative_scores = np.asarray(attribution_fn(alternative_tool), dtype=np.float32)

    if len(selected_scores) != len(tokens) or len(alternative_scores) != len(tokens):
        raise ValueError("attribution vectors must align with tokens length")

    contrastive = selected_scores - alternative_scores
    ranking = sorted(zip(tokens, contrastive.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return contrastive, ranking
