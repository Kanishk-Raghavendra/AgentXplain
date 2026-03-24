"""Token masking SHAP attribution for tool-selection confidence."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def _build_masked_tokens(tokens: Sequence[str], mask: Sequence[float], mask_token: str) -> List[str]:
    """Apply a binary mask over tokens.

    Args:
        tokens: Original token sequence.
        mask: Binary-like mask values.
        mask_token: Token used for masked positions.

    Returns:
        Masked token list.

    Raises:
        ValueError: If lengths mismatch.
    """
    if len(tokens) != len(mask):
        raise ValueError("tokens and mask must have identical length")
    return [tok if m >= 0.5 else mask_token for tok, m in zip(tokens, mask)]


def token_shap_attribution(
    tokens: Sequence[str],
    tool_score_fn,
    selected_tool: str,
    mask_token: str = "[MASK]",
    max_evals: int = 256,
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Compute token SHAP values for selected tool log-probability.

    Args:
        tokens: Token sequence.
        tool_score_fn: Callable mapping text string to tool-score dictionary.
        selected_tool: Chosen tool name.
        mask_token: Replacement token for masked positions.
        max_evals: Maximum SHAP evaluations.

    Returns:
        Tuple of SHAP values [seq_len] and ranked token-score list.

    Raises:
        ImportError: If shap is not installed.
    """
    if shap is None:
        raise ImportError("shap must be installed for token_shap_attribution")

    token_list = list(tokens)
    n = len(token_list)
    if n == 0:
        return np.array([], dtype=np.float32), []

    background = np.zeros((1, n), dtype=np.float32)

    def predict(masks: np.ndarray) -> np.ndarray:
        """Model wrapper used by SHAP explainer.

        Args:
            masks: Binary mask matrix [batch, seq_len].

        Returns:
            Array of selected tool scores [batch].

        Raises:
            KeyError: If selected tool is missing from score dictionary.
        """
        out: List[float] = []
        for row in masks:
            masked = _build_masked_tokens(token_list, row.tolist(), mask_token)
            text = " ".join(masked)
            score_map = tool_score_fn(text)
            out.append(float(score_map[selected_tool]))
        return np.asarray(out, dtype=np.float32)

    explainer = shap.Explainer(predict, background)
    shap_values = explainer(np.ones((1, n), dtype=np.float32), max_evals=max_evals)
    values = np.asarray(shap_values.values[0], dtype=np.float32)

    ranking = sorted(zip(token_list, values.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return values, ranking


def mock_token_shap(
    tokens: Sequence[str],
    keyword_set: Sequence[str],
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Generate deterministic SHAP-like values for fast local tests.

    Args:
        tokens: Token sequence.
        keyword_set: Keywords to upweight.

    Returns:
        Tuple of pseudo-SHAP values and ranked token-score list.

    Raises:
        None.
    """
    keys = {k.lower() for k in keyword_set}
    raw = np.asarray([1.2 if t.lower() in keys else -0.1 for t in tokens], dtype=np.float32)
    ranking = sorted(zip(tokens, raw.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return raw, ranking
