"""Attention rollout attribution implementation."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _fuse_heads(attn: np.ndarray, strategy: str) -> np.ndarray:
    """Fuse multi-head attention into a single matrix.

    Args:
        attn: Attention tensor with shape [heads, seq, seq].
        strategy: Head fusion strategy in {"mean", "max", "min"}.

    Returns:
        Head-fused matrix with shape [seq, seq].

    Raises:
        ValueError: If strategy is unsupported.
    """
    if strategy == "mean":
        return attn.mean(axis=0)
    if strategy == "max":
        return attn.max(axis=0)
    if strategy == "min":
        return attn.min(axis=0)
    raise ValueError("strategy must be one of: mean, max, min")


def _discard_noise(matrix: np.ndarray, discard_ratio: float) -> np.ndarray:
    """Discard low attention entries by thresholding.

    Args:
        matrix: Layer matrix [seq, seq].
        discard_ratio: Ratio of lowest values to set to zero.

    Returns:
        Thresholded matrix with same shape.

    Raises:
        ValueError: If discard_ratio is out of range.
    """
    if not 0.0 <= discard_ratio < 1.0:
        raise ValueError("discard_ratio must satisfy 0 <= discard_ratio < 1")
    if discard_ratio == 0.0:
        return matrix

    flat = matrix.flatten()
    k = int(discard_ratio * flat.size)
    if k <= 0:
        return matrix

    threshold = np.partition(flat, k)[k]
    out = matrix.copy()
    out[out < threshold] = 0.0
    return out


def compute_rollout(
    attentions: Sequence[np.ndarray],
    head_fusion: str = "mean",
    discard_ratio: float = 0.1,
) -> np.ndarray:
    """Compute Abnar and Zuidema attention rollout.

    Args:
        attentions: Sequence of layer attentions [batch, heads, seq, seq].
        head_fusion: Head fusion strategy.
        discard_ratio: Ratio used for low-attention suppression.

    Returns:
        Rollout matrix [seq, seq].

    Raises:
        ValueError: If attentions are empty or malformed.
    """
    if not attentions:
        raise ValueError("attentions cannot be empty")

    first = np.asarray(attentions[0])
    if first.ndim != 4:
        raise ValueError("each attention tensor must have shape [batch, heads, seq, seq]")

    seq_len = int(first.shape[-1])
    joint = np.eye(seq_len, dtype=np.float32)

    for layer in attentions:
        arr = np.asarray(layer, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError("each attention tensor must have shape [batch, heads, seq, seq]")

        fused = _fuse_heads(arr[0], head_fusion)
        fused = _discard_noise(fused, discard_ratio)

        fused = fused + np.eye(seq_len, dtype=np.float32)
        fused = fused / (fused.sum(axis=-1, keepdims=True) + 1e-12)
        joint = fused @ joint

    return joint


def attention_rollout_attribution(
    attentions: Sequence[np.ndarray],
    tokens: Sequence[str],
    head_fusion: str = "mean",
    discard_ratio: float = 0.1,
    target_position: int | None = None,
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Generate rollout matrix and ranked token scores.

    Args:
        attentions: Sequence of layer attentions [batch, heads, seq, seq].
        tokens: Tokens aligned with sequence positions.
        head_fusion: Head fusion strategy.
        discard_ratio: Ratio used for low-attention suppression.
        target_position: Row index to interpret, default uses last token.

    Returns:
        Tuple of rollout matrix and ranked (token, score) list.

    Raises:
        ValueError: If token length and matrix dimensions mismatch.
    """
    rollout = compute_rollout(attentions, head_fusion=head_fusion, discard_ratio=discard_ratio)
    if len(tokens) != rollout.shape[0]:
        raise ValueError("tokens length must equal rollout dimension")

    idx = rollout.shape[0] - 1 if target_position is None else int(target_position)
    scores = rollout[idx]
    ranking = sorted(zip(tokens, scores.tolist()), key=lambda x: x[1], reverse=True)
    return rollout, ranking
