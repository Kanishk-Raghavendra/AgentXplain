"""Agent tests for AgentXplain."""

from __future__ import annotations

from unittest.mock import patch

from src.agent.agent import AgentXplainAgent
from src.agent.tools import calculator, code_executor, web_search


def test_mock_tools_outputs() -> None:
    """Verify deterministic mock tool behavior.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    assert web_search("weather today") == "[Search result for: weather today]"
    assert calculator("2 + 3 * 4") == "14"
    assert "stdout: hi" in code_executor("print('hi')")


def test_agent_trace_fields() -> None:
    """Ensure trace dataclass fields are populated.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    agent = AgentXplainAgent(use_mock_router=True)
    trace = agent.run("please calculate 9+3")
    assert trace.query
    assert trace.selected_tool in {"web_search", "calculator", "code_executor"}
    assert isinstance(trace.args, dict)
    assert isinstance(trace.reason, str)
    assert isinstance(trace.input_token_ids, list)
    assert isinstance(trace.attention_weights, list)
    assert isinstance(trace.tool_score_distribution, dict)


def test_tool_parsing_in_mock_mode() -> None:
    """Check heuristic tool parsing in mock mode.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    agent = AgentXplainAgent(use_mock_router=True)
    assert agent.run("calculate 8*7").selected_tool == "calculator"
    assert agent.run("execute python loop").selected_tool == "code_executor"
    assert agent.run("who is the current prime minister").selected_tool == "web_search"


def test_agent_run_uses_internal_router() -> None:
    """Ensure run delegates to internal mock router in mock mode.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    agent = AgentXplainAgent(use_mock_router=True)
    with patch.object(agent, "_mock_route", wraps=agent._mock_route) as mocked:
        _ = agent.run("calculate 1+1")
        mocked.assert_called_once()
