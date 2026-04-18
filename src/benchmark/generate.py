"""Synthetic benchmark generation for AgentXplain causal attribution evaluation."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class BenchmarkTrace:
    """One synthetic trace with known trigger spans.

    Args:
        id: Numeric sample identifier.
        query: Input query text.
        planted_trigger: Trigger phrase expected to indicate correct tool.
        correct_tool: Ground-truth tool name.
        distractor_keywords: Distractor keywords from other domains.
        ground_truth_arg_hint: Hint for expected argument extraction.
        split: Dataset split label in {"standard", "hard", "paraphrase", "negation"}.
        paraphrase_of: Id of original trace if this is a paraphrase trace, else None.
    """

    id: int
    query: str
    planted_trigger: str
    correct_tool: str
    distractor_keywords: List[str]
    ground_truth_arg_hint: str
    split: str
    paraphrase_of: Optional[int]


TOOL_SPECS: Dict[str, Dict[str, Sequence[str]]] = {
    "calculator": {
        "triggers": ["calculate", "sum", "multiply", "divide", "arithmetic", "evaluate", "compute", "equation"],
        "paraphrases": ["find the product", "work out the value", "determine the numeric result", "compute mentally"],
        "templates": [
            "Please {trigger} {hint}. Also find context about {d1}.",
            "I need you to {trigger} this expression: {hint}; unrelated terms: {d1}, {d2}.",
            "Use arithmetic to {trigger} {hint} while skipping {d1}.",
            "Math request with trigger {trigger}: {hint}; distractor {d1} and {d2}.",
            "Can you {trigger} result for {hint}? This is not about {d1}.",
            "Return numeric answer for {hint}; keyword is {trigger}; noisy words {d1}/{d2}.",
            "Only do arithmetic: {trigger} {hint}. Ignore {d1}.",
            "Task: {trigger} {hint}; if needed ignore {d1} and {d2}.",
        ],
        "hints": ["23 + 19", "81/9", "14*7", "(45-17)*2", "6**2 + 5", "(18+6)/3", "9*9-4", "120/5+7"],
        "negation": [
            "Do NOT calculate this, just tell me approximately: 145 * 37",
            "Without doing arithmetic, guess the result of 84 / 6",
            "Do not compute exactly; provide intuition for 19*12",
        ],
    },
    "web_search": {
        "triggers": ["search", "latest", "news", "who", "when", "where", "lookup", "find"],
        "paraphrases": ["look up", "retrieve information", "check current updates", "gather recent facts"],
        "templates": [
            "Please {trigger} information about {hint}; note this number {d1}.",
            "Need factual lookup: {trigger} {hint}. Distractors: {d1}, {d2}.",
            "Can you {trigger} for {hint} and avoid {d1}?",
            "Knowledge query with signal {trigger}: {hint}; noise {d1}/{d2}.",
            "Find external info on {hint}; trigger word {trigger}; avoid {d1}.",
            "I want current facts: {trigger} {hint}. Unrelated: {d1}, {d2}.",
            "Use web retrieval to answer: {hint}. Cue is {trigger}.",
            "Question: {hint}. Please {trigger}, not {d1}.",
        ],
        "hints": [
            "the capital of Bhutan",
            "the latest Mars mission",
            "who discovered penicillin",
            "when the internet was invented",
            "where COP summits are hosted",
            "news about quantum chips",
            "latest olympics host city",
            "find global inflation report",
        ],
        "negation": [
            "Without searching the web, what do you know about climate change?",
            "Do not look anything up online; answer from memory about Jupiter",
            "Please avoid web search and just provide your internal summary of AI safety",
        ],
    },
    "code_executor": {
        "triggers": ["python", "execute", "script", "loop", "function", "print", "run code", "debug"],
        "paraphrases": ["write a routine", "implement a procedure", "program the solution", "compose executable steps"],
        "templates": [
            "Please {trigger} this task: {hint}. Also calculate this mentally: {d1}.",
            "Write and run code to {hint}; distractors {d1}, {d2}.",
            "I need a Python solution: {trigger} for {hint}; not {d1}.",
            "Programming request with token {trigger}: {hint}; avoid {d2}.",
            "Could you {trigger} and show output for {hint}?",
            "Code-only task: {hint}. Signal={trigger}. Noise={d1}/{d2}.",
            "Use code execution for {hint}; irrelevant keyword {d1}.",
            "Implement and run: {hint}; instruction trigger {trigger}.",
        ],
        "hints": [
            "print numbers 1 to 5",
            "define a factorial function",
            "sum a list in Python",
            "show a for loop with conditionals",
            "print hello world twice",
            "sort an array and print it",
            "write a function for fibonacci",
            "execute a small dictionary demo",
        ],
        "negation": [
            "Explain the logic without writing any code: how would you sort a list?",
            "Do not execute code, just describe how to implement binary search",
            "Without coding, explain a loop that sums numbers 1 to 10",
        ],
    },
}

TOOL_FALLBACK = "none"

DISTRACTOR_POOLS: Dict[str, Sequence[str]] = {
    "calculator": [
        "calculate",
        "multiply",
        "multiplied",
        "divide",
        "sum",
        "percent",
        "times",
        "plus",
        "minus",
        "compute",
    ],
    "web_search": [
        "search",
        "find",
        "lookup",
        "latest",
        "news",
        "current",
        "who",
        "what",
    ],
    "code_executor": [
        "code",
        "script",
        "function",
        "python",
        "program",
        "implement",
        "algorithm",
    ],
}


def _split_counts(n: int) -> Dict[str, int]:
    """Compute split counts for standard, hard, paraphrase, and negation.

    Args:
        n: Total requested trace count.

    Returns:
        Mapping split->count summing to n.

    Raises:
        ValueError: If n is less than 4.
    """
    if n < 4:
        raise ValueError("n must be at least 4 to populate all splits")

    base = {
        "standard": int(round(n * 0.5)),
        "hard": int(round(n / 6)),
        "paraphrase": int(round(n / 6)),
        "negation": int(round(n / 6)),
    }
    delta = n - sum(base.values())
    order = ["standard", "hard", "paraphrase", "negation"]
    idx = 0
    while delta != 0:
        key = order[idx % len(order)]
        if delta > 0:
            base[key] += 1
            delta -= 1
        elif base[key] > 1:
            base[key] -= 1
            delta += 1
        idx += 1
    return base


def _sample_distractors(tool: str, rng: random.Random) -> List[str]:
    """Sample cross-domain distractor keywords.

    Args:
        tool: Correct tool domain.
        rng: Random generator.

    Returns:
        List of two distractor keywords.

    Raises:
        ValueError: If tool is unknown.
    """
    if tool not in TOOL_SPECS:
        raise ValueError(f"Unknown tool: {tool}")

    allowed_domains = [domain for domain in DISTRACTOR_POOLS if domain != tool]
    candidates: List[str] = []
    for domain in allowed_domains:
        candidates.extend(DISTRACTOR_POOLS[domain])

    if len(candidates) < 2:
        raise ValueError("insufficient cross-domain distractor candidates")

    return rng.sample(candidates, 2)


def _hard_lexical_distractors(tool: str) -> List[str]:
    """Return distractors lexically similar to other tool triggers.

    Args:
        tool: Correct tool domain.

    Returns:
        Two lexically confusable distractors.

    Raises:
        ValueError: If tool is unknown.
    """
    mapping = {
        "calculator": ["search keyword", "python function"],
        "web_search": ["compute value", "algorithm script"],
        "code_executor": ["latest lookup", "calculate sum"],
    }
    if tool not in mapping:
        raise ValueError(f"Unknown tool: {tool}")
    return mapping[tool]


def _build_trace(
    idx: int,
    split: str,
    tool: str,
    trigger: str,
    hint: str,
    distractors: List[str],
    query: str,
    paraphrase_of: Optional[int],
) -> BenchmarkTrace:
    """Create one benchmark trace.

    Args:
        idx: Trace id.
        split: Split name.
        tool: Correct tool label.
        trigger: Planted trigger concept.
        hint: Ground-truth argument hint.
        distractors: Distractor keywords.
        query: Query text.
        paraphrase_of: Source trace id for paraphrases.

    Returns:
        BenchmarkTrace instance.

    Raises:
        None.
    """
    return BenchmarkTrace(
        id=idx,
        query=query,
        planted_trigger=trigger,
        correct_tool=tool,
        distractor_keywords=distractors,
        ground_truth_arg_hint=hint,
        split=split,
        paraphrase_of=paraphrase_of,
    )


def _ensure_trigger_in_query(trigger: str, query: str) -> str:
    """Ensure the exact trigger phrase appears as a substring in query text."""
    if trigger.lower() in query.lower():
        return query
    return f"{query} Trigger phrase: {trigger}."


def generate_benchmark(n: int = 300, seed: int = 42) -> List[BenchmarkTrace]:
    """Generate synthetic benchmark traces.

    Args:
        n: Number of traces.
        seed: Random seed for reproducibility.

    Returns:
        List of BenchmarkTrace entries.

    Raises:
        ValueError: If n is not positive.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = random.Random(seed)
    tools = list(TOOL_SPECS.keys())
    split_counts = _split_counts(n)
    traces: List[BenchmarkTrace] = []

    idx = 0
    for split in ["standard", "hard", "paraphrase", "negation"]:
        for _ in range(split_counts[split]):
            tool = tools[idx % len(tools)]
            spec = TOOL_SPECS[tool]
            trigger = rng.choice(list(spec["triggers"]))
            hint = rng.choice(list(spec["hints"]))
            distractors = _sample_distractors(tool, rng)

            if split == "hard":
                distractors = _hard_lexical_distractors(tool)
                template = rng.choice(list(spec["templates"]))
                query = template.format(trigger=trigger, hint=hint, d1=distractors[0], d2=distractors[1])
                query = _ensure_trigger_in_query(trigger, query)
                trace = _build_trace(idx, split, tool, trigger, hint, distractors, query, paraphrase_of=None)
            elif split == "paraphrase":
                base_id = max(0, idx - 1)
                phrase = rng.choice(list(spec["paraphrases"]))
                if tool == "calculator":
                    query = f"{phrase} of {hint}; this is not a web lookup request."
                elif tool == "web_search":
                    query = f"I need current information regarding {hint}; avoid direct calculation of 145 * 37."
                else:
                    query = f"{phrase} in Python that handles: {hint}; mention calculate only as context."
                query = _ensure_trigger_in_query(trigger, query)
                trace = _build_trace(idx, split, tool, trigger, hint, distractors, query, paraphrase_of=base_id)
            elif split == "negation":
                query = rng.choice(list(spec["negation"]))
                query = _ensure_trigger_in_query(trigger, query)
                trace = _build_trace(
                    idx,
                    split,
                    TOOL_FALLBACK,
                    trigger,
                    hint,
                    distractors,
                    query,
                    paraphrase_of=None,
                )
            else:
                template = rng.choice(list(spec["templates"]))
                query = template.format(trigger=trigger, hint=hint, d1=distractors[0], d2=distractors[1])
                query = _ensure_trigger_in_query(trigger, query)
                trace = _build_trace(idx, split, tool, trigger, hint, distractors, query, paraphrase_of=None)

            traces.append(trace)
            idx += 1

    return traces


def save_benchmark(traces: Sequence[BenchmarkTrace], output_path: Path) -> None:
    """Save benchmark traces to one JSON file.

    Args:
        traces: Sequence of benchmark traces.
        output_path: Destination file path.

    Returns:
        None.

    Raises:
        OSError: If writing fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(trace) for trace in traces]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_split_benchmarks(traces: Sequence[BenchmarkTrace], output_dir: Path) -> None:
    """Save split-wise and full benchmark JSON files.

    Args:
        traces: Combined benchmark traces.
        output_dir: Directory for benchmark JSON files.

    Returns:
        None.

    Raises:
        OSError: If writing fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    split_map = {
        "standard": [],
        "hard": [],
        "paraphrase": [],
        "negation": [],
    }
    for trace in traces:
        if trace.split in split_map:
            split_map[trace.split].append(asdict(trace))

    (output_dir / "benchmark_standard.json").write_text(
        json.dumps(split_map["standard"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark_hard.json").write_text(
        json.dumps(split_map["hard"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark_paraphrase.json").write_text(
        json.dumps(split_map["paraphrase"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark_negation.json").write_text(
        json.dumps(split_map["negation"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark_full.json").write_text(
        json.dumps([asdict(trace) for trace in traces], indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse benchmark generation command-line arguments.

    Args:
        None.

    Returns:
        Parsed argparse namespace.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description="Generate AgentXplain synthetic benchmark")
    parser.add_argument("--n", type=int, default=300, help="Number of synthetic traces")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/synthetic/benchmark_full.json"),
        help="Output full benchmark JSON path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Directory where split benchmark JSON files are written",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for benchmark generation.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    args = parse_args()
    traces = generate_benchmark(n=args.n, seed=args.seed)
    out_path = args.out
    out_dir = args.out_dir

    # Support `--out data/synthetic` by treating it as the benchmark directory.
    if out_path.exists() and out_path.is_dir():
        out_dir = out_path
        out_path = out_dir / "benchmark_full.json"
    elif out_path.suffix.lower() != ".json":
        out_dir = out_path
        out_path = out_dir / "benchmark_full.json"

    save_split_benchmarks(traces, out_dir)
    save_benchmark(traces, out_path)
    print(f"Saved {len(traces)} traces to {out_path}")


if __name__ == "__main__":
    main()
