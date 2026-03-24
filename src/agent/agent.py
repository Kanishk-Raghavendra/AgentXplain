"""Tool-routing agent with structured output and trace capture."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


TOOL_NAMES = ["web_search", "calculator", "code_executor"]


@dataclass
class AgentTrace:
    """Trace data for one tool-selection decision.

    Args:
        query: Original user query.
        selected_tool: Selected tool name.
        args: Parsed tool arguments.
        reason: Parsed reason string.
        input_token_ids: Input token ids for the decision prompt.
        attention_weights: Attention tensors per layer [batch, heads, seq, seq].
        tool_score_distribution: Logit score per tool at decision position.
    """

    query: str
    selected_tool: str
    args: Dict[str, Any]
    reason: str
    input_token_ids: List[int]
    attention_weights: List[List[List[List[float]]]]
    tool_score_distribution: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the trace to a JSON-serializable dictionary.

        Args:
            None.

        Returns:
            Dictionary representation of AgentTrace.

        Raises:
            None.
        """
        return asdict(self)


class AgentXplainAgent:
    """AgentXplain routing agent wrapper around Hugging Face models."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: Optional[str] = None,
        use_mock_router: bool = False,
    ) -> None:
        """Initialize the agent.

        Args:
            model_name: Hugging Face model identifier.
            device: Torch device string.
            use_mock_router: If True, use deterministic local heuristics.

        Returns:
            None.

        Raises:
            RuntimeError: If model loading fails and mock mode is disabled.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mock_router = use_mock_router
        self.tokenizer = None
        self.model = None

        if self.use_mock_router:
            return

        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is required unless use_mock_router=True")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, query: str) -> str:
        """Build the routing prompt with strict structured output format.

        Args:
            query: User query string.

        Returns:
            Prompt string.

        Raises:
            ValueError: If query is empty.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")

        return (
            "You are a strict tool router. Choose one tool among "
            "web_search, calculator, code_executor.\\n"
            "Return exactly this format and nothing else:\\n"
            "TOOL: <tool_name>\\n"
            "ARGS: <json_object>\\n"
            "REASON: <short_reason>\\n"
            "Tool descriptions:\\n"
            "- web_search(query): factual lookup and current information\\n"
            "- calculator(expression): arithmetic computations\\n"
            "- code_executor(code): run short Python snippets\\n"
            f"User query: {query}\\n"
        )

    def _score_tools(self, logits: torch.Tensor) -> Dict[str, float]:
        """Map candidate tool names to logits at decision position.

        Args:
            logits: Next-token logits tensor [vocab].

        Returns:
            Mapping from tool name to score.

        Raises:
            RuntimeError: If tokenizer is unavailable.
        """
        if self.tokenizer is None:
            raise RuntimeError("tokenizer not initialized")

        score_map: Dict[str, float] = {}
        for tool in TOOL_NAMES:
            tool_tokens = self.tokenizer(tool, add_special_tokens=False)["input_ids"]
            first_token = int(tool_tokens[0]) if tool_tokens else int(self.tokenizer.eos_token_id)
            score_map[tool] = float(logits[first_token].item())
        return score_map

    def _parse_structured_output(self, text: str, query: str) -> Tuple[str, Dict[str, Any], str]:
        """Parse TOOL/ARGS/REASON blocks from model output.

        Args:
            text: Decoded model response.
            query: Original query for fallback routing.

        Returns:
            Tuple of selected tool, args dictionary, and reason.

        Raises:
            None.
        """
        tool_match = re.search(r"TOOL:\s*(.+)", text)
        args_match = re.search(r"ARGS:\s*(.+)", text)
        reason_match = re.search(r"REASON:\s*(.+)", text)

        tool = tool_match.group(1).strip() if tool_match else "web_search"
        if tool not in TOOL_NAMES:
            tool = "web_search"

        args: Dict[str, Any] = {"query": query}
        if args_match:
            raw = args_match.group(1).strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {"query": query}

        if tool == "calculator" and "expression" not in args:
            args = {"expression": "1+1"}
        if tool == "code_executor" and "code" not in args:
            args = {"code": "print('hello')"}
        if tool == "web_search" and "query" not in args:
            args = {"query": query}

        reason = reason_match.group(1).strip() if reason_match else "No reason provided."
        return tool, args, reason

    def _mock_route(self, query: str) -> AgentTrace:
        """Run deterministic mock routing for CPU-only testing.

        Args:
            query: User query.

        Returns:
            AgentTrace with heuristic routing outputs.

        Raises:
            None.
        """
        q = query.lower()
        if any(token in q for token in ["calculate", "sum", "multiply", "divide", "+", "-", "*", "/"]):
            tool = "calculator"
            args = {"expression": "".join(ch for ch in query if ch.isdigit() or ch in "+-*/(). ") or "1+1"}
            reason = "Arithmetic intent detected."
            scores = {"calculator": 1.0, "web_search": 0.2, "code_executor": 0.1}
        elif any(token in q for token in ["code", "python", "script", "execute", "function"]):
            tool = "code_executor"
            args = {"code": "print('mock execution')"}
            reason = "Programming intent detected."
            scores = {"calculator": 0.1, "web_search": 0.2, "code_executor": 1.0}
        else:
            tool = "web_search"
            args = {"query": query}
            reason = "Information lookup intent detected."
            scores = {"calculator": 0.2, "web_search": 1.0, "code_executor": 0.1}

        token_ids = [ord(ch) % 255 for ch in query][:128]
        seq = max(1, len(token_ids))
        eye = torch.eye(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(0).tolist()

        return AgentTrace(
            query=query,
            selected_tool=tool,
            args=args,
            reason=reason,
            input_token_ids=token_ids,
            attention_weights=[eye],
            tool_score_distribution=scores,
        )

    def run(self, query: str) -> AgentTrace:
        """Run routing for a single query.

        Args:
            query: User query string.

        Returns:
            AgentTrace containing structured routing evidence.

        Raises:
            RuntimeError: If model components are unavailable in non-mock mode.
        """
        if self.use_mock_router:
            return self._mock_route(query)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model/tokenizer not initialized")

        prompt = self._build_prompt(query)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            forward = self.model(**encoded, output_attentions=True)
            decision_logits = forward.logits[0, -1, :]
            tool_scores = self._score_tools(decision_logits)

            generated = self.model.generate(
                **encoded,
                max_new_tokens=100,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        sequence = generated.sequences[0]
        input_len = encoded["input_ids"].shape[1]
        new_tokens = sequence[input_len:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        selected_tool, args, reason = self._parse_structured_output(decoded, query)

        attentions: List[List[List[List[float]]]] = []
        if generated.attentions and len(generated.attentions) > 0:
            for layer_tensor in generated.attentions[-1]:
                attentions.append(layer_tensor.detach().cpu().tolist())

        return AgentTrace(
            query=query,
            selected_tool=selected_tool,
            args=args,
            reason=reason,
            input_token_ids=encoded["input_ids"][0].detach().cpu().tolist(),
            attention_weights=attentions,
            tool_score_distribution=tool_scores,
        )
