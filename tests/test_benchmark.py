"""Benchmark tests for AgentXplain."""

from __future__ import annotations

from src.benchmark.generate import generate_benchmark


def test_count_and_tool_balance() -> None:
    """Benchmark should include all tools and correct count.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    traces = generate_benchmark(n=300, seed=11)
    assert len(traces) == 300
    tools = {trace.correct_tool for trace in traces}
    assert {"calculator", "web_search", "code_executor", "none"}.issubset(tools)


def test_reproducibility() -> None:
    """Benchmark generation should be deterministic with seed.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    a = generate_benchmark(n=30, seed=12)
    b = generate_benchmark(n=30, seed=12)
    assert [x.query for x in a] == [x.query for x in b]


def test_required_splits_present() -> None:
    """Generated benchmark should include hard, paraphrase, and negation splits.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    traces = generate_benchmark(n=300, seed=9)
    splits = {trace.split for trace in traces}
    assert "hard" in splits
    assert "paraphrase" in splits
    assert "negation" in splits


def test_split_field_populated_for_all_traces() -> None:
    """Every generated trace should have a populated split field.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    traces = generate_benchmark(n=300, seed=5)
    for trace in traces:
        assert isinstance(trace.split, str)
        assert trace.split in {"standard", "hard", "paraphrase", "negation"}
