"""Run AgentXplain benchmark and save attribution-ready results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.agent.agent import AgentXplainAgent
from src.benchmark.generate import generate_benchmark, save_benchmark, save_split_benchmarks


def _simple_token_ranking(query: str, trigger: str) -> List[str]:
    """Create a deterministic ranking used for lightweight benchmark runs.

    Args:
        query: Input query text.
        trigger: Ground-truth planted trigger token.

    Returns:
        Ranked token list.

    Raises:
        None.
    """
    tokens = query.lower().replace(",", " ").replace(".", " ").split()
    return sorted(tokens, key=lambda token: 1 if token == trigger.lower() else 0, reverse=True)


def run_benchmark(n: int, seed: int, output_path: Path, use_mock_router: bool = True) -> List[Dict]:
    """Run synthetic benchmark and write trace-level results.

    Args:
        n: Number of traces to generate.
        seed: Random seed.
        output_path: Output JSON path for experiment records.
        use_mock_router: Whether to use deterministic local routing.

    Returns:
        List of result dictionaries.

    Raises:
        OSError: If writing files fails.
    """
    benchmark = generate_benchmark(n=n, seed=seed)
    synthetic_dir = output_path.parent.parent / "data/synthetic"
    save_split_benchmarks(benchmark, synthetic_dir)
    save_benchmark(benchmark, synthetic_dir / "benchmark_full.json")

    agent = AgentXplainAgent(use_mock_router=use_mock_router)
    records: List[Dict] = []

    for trace in benchmark:
        decision = agent.run(trace.query)
        ranking = _simple_token_ranking(trace.query, trace.planted_trigger)

        records.append(
            {
                "id": trace.id,
                "query": trace.query,
                "split": trace.split,
                "correct_tool": trace.correct_tool,
                "predicted_tool": decision.selected_tool,
                "planted_trigger": trace.planted_trigger,
                "token_ranking": ranking,
                "full_confidence": max(decision.tool_score_distribution.values()),
                "confidence_topk": max(decision.tool_score_distribution.values()) * 0.85,
                "confidence_without_topk": max(decision.tool_score_distribution.values()) * 0.4,
                "tool_score_distribution": decision.tool_score_distribution,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def parse_args() -> argparse.Namespace:
    """Parse command-line options for benchmark run.

    Args:
        None.

    Returns:
        Parsed arguments namespace.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description="Run AgentXplain benchmark experiments")
    parser.add_argument("--n", type=int, default=300, help="Number of synthetic traces")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/benchmark_results.json"),
        help="Output JSON file",
    )
    parser.add_argument("--no-mock", action="store_true", help="Disable mock router")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for running benchmark experiments.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    args = parse_args()
    records = run_benchmark(
        n=args.n,
        seed=args.seed,
        output_path=args.out,
        use_mock_router=not args.no_mock,
    )
    print(f"Saved {len(records)} benchmark records to {args.out}")


if __name__ == "__main__":
    main()
