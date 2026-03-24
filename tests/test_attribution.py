"""Attribution tests for AgentXplain."""

from __future__ import annotations

import numpy as np

from src.attribution.attention_rollout import attention_rollout_attribution
from src.attribution.contrastive import contrastive_attribution
from src.attribution.gradient_saliency import mock_gradient_saliency
from src.attribution.token_shap import mock_token_shap


def test_attention_rollout_shapes() -> None:
    """Validate rollout and ranking output shape.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    seq = 7
    layer = np.ones((1, 2, seq, seq), dtype=np.float32)
    tokens = [f"t{i}" for i in range(seq)]
    rollout, ranking = attention_rollout_attribution([layer, layer], tokens, discard_ratio=0.0)
    assert rollout.shape == (seq, seq)
    assert len(ranking) == seq


def test_gradient_and_shap_mock_lengths() -> None:
    """Validate mock saliency lengths and normalization range.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    tokens = ["please", "calculate", "23", "+", "19"]
    grad_scores, grad_rank = mock_gradient_saliency(tokens, ["calculate", "23", "19"])
    shap_scores, shap_rank = mock_token_shap(tokens, ["calculate", "23", "19"])

    assert len(grad_scores) == len(tokens)
    assert len(grad_rank) == len(tokens)
    assert float(grad_scores.min()) >= 0.0
    assert float(grad_scores.max()) <= 1.0
    assert len(shap_scores) == len(tokens)
    assert len(shap_rank) == len(tokens)


def test_contrastive_output_length() -> None:
    """Ensure contrastive saliency aligns with tokens.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    tokens = ["a", "b", "c"]

    def _fake_method(tool_name: str) -> np.ndarray:
        return np.array([1.0, 0.4, 0.2], dtype=np.float32) if tool_name == "calculator" else np.array([0.2, 0.5, 0.1], dtype=np.float32)

    contrastive, ranking = contrastive_attribution(_fake_method, tokens, "calculator", "web_search")
    assert len(contrastive) == len(tokens)
    assert len(ranking) == len(tokens)
    assert any(score > 0 for score in contrastive.tolist())
    assert any(score < 0 for score in contrastive.tolist())
