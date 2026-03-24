"""Faithfulness probe tests for AgentXplain."""

from __future__ import annotations

import pytest

from src.faithfulness.probes import comprehensiveness, sufficiency


class DummyModel:
    """Simple confidence model for faithfulness tests."""

    def tool_confidence(self, token_ids):
        """Return bounded confidence based on token count.

        Args:
            token_ids: Token ids.

        Returns:
            Confidence in [0, 1].

        Raises:
            None.
        """
        return min(1.0, max(0.0, len(token_ids) / 10.0))


def test_sufficiency_range() -> None:
    """Sufficiency should stay in [0, 1].

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    model = DummyModel()
    token_ids = [1, 2, 3, 4, 5]
    mask = [True, False, True, False, False]
    score = sufficiency(model, token_ids, mask)
    assert 0.0 <= score <= 1.0


def test_comprehensiveness_range() -> None:
    """Comprehensiveness should stay in [0, 1].

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    model = DummyModel()
    token_ids = [1, 2, 3, 4, 5]
    mask = [True, False, True, False, False]
    score = comprehensiveness(model, token_ids, mask)
    assert 0.0 <= score <= 1.0


def test_mask_length_mismatch_raises() -> None:
    """Length mismatch should raise ValueError.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    model = DummyModel()
    with pytest.raises(ValueError):
        sufficiency(model, [1, 2, 3], [True, False])
