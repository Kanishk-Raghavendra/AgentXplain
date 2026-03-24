"""Visualization utilities for token attributions."""

from __future__ import annotations

from html import escape
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def _normalize(scores: Sequence[float]) -> np.ndarray:
    """Normalize numeric scores to [0, 1].

    Args:
        scores: Sequence of score values.

    Returns:
        Normalized numpy array.

    Raises:
        None.
    """
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr
    min_v = float(arr.min())
    max_v = float(arr.max())
    if abs(max_v - min_v) < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def token_highlight_html(tokens: Sequence[str], scores: Sequence[float]) -> str:
    """Create HTML with attribution-colored token spans.

    Args:
        tokens: Token strings.
        scores: Attribution scores.

    Returns:
        HTML snippet string.

    Raises:
        ValueError: If lengths are inconsistent.
    """
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have same length")

    norm = _normalize(scores)
    cmap = cm.get_cmap("YlOrRd")
    parts = ["<div style='font-family:monospace;line-height:2'>"]

    for token, score in zip(tokens, norm):
        r, g, b, _ = cmap(float(score))
        rgba = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.75)"
        parts.append(
            "<span style='padding:2px 4px;margin:1px;border-radius:4px;"
            f"background:{rgba};'>{escape(token)}</span>"
        )

    parts.append("</div>")
    return "".join(parts)


def plot_token_bar(tokens: Sequence[str], scores: Sequence[float], top_k: int = 15):
    """Plot a horizontal bar chart of top-k token attributions.

    Args:
        tokens: Token strings.
        scores: Attribution scores.
        top_k: Number of top tokens to plot.

    Returns:
        Matplotlib figure and axis.

    Raises:
        ValueError: If lengths are inconsistent.
    """
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have same length")

    pairs = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)[:top_k]
    labels = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.barh(labels, values, color="#cc4c02")
    ax.set_xlabel("Attribution Score")
    ax.set_title(f"Top-{top_k} Token Attribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig, ax


def plot_attention_heatmap(matrix: np.ndarray, tokens: Sequence[str], max_tokens: int = 40):
    """Plot attention rollout heatmap for a token sequence.

    Args:
        matrix: Rollout matrix [seq, seq].
        tokens: Token sequence.
        max_tokens: Maximum displayed token count.

    Returns:
        Matplotlib figure and axis.

    Raises:
        ValueError: If matrix is not square or misaligned with tokens.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    if len(tokens) != arr.shape[0]:
        raise ValueError("token length must match matrix size")

    limit = min(max_tokens, arr.shape[0])
    clipped = arr[:limit, :limit]
    clipped_tokens = list(tokens)[:limit]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    image = ax.imshow(clipped, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(limit))
    ax.set_yticks(range(limit))
    ax.set_xticklabels(clipped_tokens, rotation=90, fontsize=7)
    ax.set_yticklabels(clipped_tokens, fontsize=7)
    ax.set_title("Attention Rollout Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax
