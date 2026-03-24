"""Dissociation analysis between tool-level and span-level attribution outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from scipy.stats import chi2_contingency


def _matrix_counts(records: List[Dict]) -> Dict[str, int]:
    """Build 2x2 dissociation counts from records.

    Args:
        records: Trace-level result dictionaries.

    Returns:
        Dictionary with quadrant counts.

    Raises:
        KeyError: If required boolean keys are absent.
    """
    counts = {
        "both_correct": 0,
        "tool_correct_span_wrong": 0,
        "tool_wrong_span_correct": 0,
        "both_wrong": 0,
    }
    for rec in records:
        tool_ok = bool(rec["agentshap_tool_correct"])
        span_ok = bool(rec["span_hit"])
        if tool_ok and span_ok:
            counts["both_correct"] += 1
        elif tool_ok and not span_ok:
            counts["tool_correct_span_wrong"] += 1
        elif (not tool_ok) and span_ok:
            counts["tool_wrong_span_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def _top_examples(records: List[Dict], tool_ok: bool, span_ok: bool, top_n: int = 10) -> List[Dict]:
    """Get top informative examples for a dissociation quadrant.

    Args:
        records: Trace-level result dictionaries.
        tool_ok: Desired tool-level correctness.
        span_ok: Desired span-level correctness.
        top_n: Maximum number of examples to return.

    Returns:
        List of concise example dictionaries.

    Raises:
        None.
    """
    selected = [
        rec for rec in records
        if bool(rec.get("agentshap_tool_correct", False)) == tool_ok
        and bool(rec.get("span_hit", False)) == span_ok
    ]
    ranked = sorted(
        selected,
        key=lambda r: float(r.get("informativeness", r.get("confidence_gap", 0.0))),
        reverse=True,
    )
    out: List[Dict] = []
    for rec in ranked[:top_n]:
        out.append(
            {
                "id": rec.get("id"),
                "query": rec.get("query"),
                "split": rec.get("split"),
                "correct_tool": rec.get("correct_tool"),
                "predicted_tool": rec.get("predicted_tool"),
                "planted_trigger": rec.get("planted_trigger"),
            }
        )
    return out


def analyze_dissociation(records: List[Dict]) -> Dict[str, object]:
    """Compute dissociation matrix, chi-squared test, and key examples.

    Args:
        records: Trace-level result dictionaries.

    Returns:
        Summary dictionary ready for JSON serialization.

    Raises:
        ValueError: If records are empty.
    """
    if not records:
        raise ValueError("records cannot be empty")

    counts = _matrix_counts(records)
    matrix = [
        [counts["both_correct"], counts["tool_correct_span_wrong"]],
        [counts["tool_wrong_span_correct"], counts["both_wrong"]],
    ]

    chi2, p_value, _, _ = chi2_contingency(matrix)
    key_tool_correct_span_wrong = _top_examples(records, tool_ok=True, span_ok=False, top_n=10)
    key_tool_wrong_span_correct = _top_examples(records, tool_ok=False, span_ok=True, top_n=10)

    interpretation = (
        "The two attribution levels appear statistically dependent."
        if p_value < 0.05
        else "The two attribution levels appear statistically independent, supporting non-redundancy."
    )

    return {
        "confusion_matrix": {
            "rows": "span_hit [True, False]",
            "cols": "tool_correct [True, False]",
            "matrix": matrix,
        },
        "quadrants": counts,
        "chi_squared": {
            "statistic": float(chi2),
            "p_value": float(p_value),
            "interpretation": interpretation,
        },
        "top_tool_correct_span_wrong": key_tool_correct_span_wrong,
        "top_tool_wrong_span_correct": key_tool_wrong_span_correct,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dissociation analysis.

    Args:
        None.

    Returns:
        Parsed namespace.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description="Run AgentXplain dissociation analysis")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to input results JSON with agentshap_tool_correct and span_hit fields",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("results/dissociation_summary.json"),
        help="Path to save dissociation summary JSON",
    )
    parser.add_argument("--verbose", action="store_true", help="Print top examples")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for dissociation analysis.

    Args:
        None.

    Returns:
        None.

    Raises:
        OSError: If file read/write fails.
    """
    args = parse_args()
    records = json.loads(args.results.read_text(encoding="utf-8"))
    summary = analyze_dissociation(records)

    matrix = summary["confusion_matrix"]["matrix"]
    chi = summary["chi_squared"]

    print("=== Dissociation Matrix (rows: span_hit, cols: tool_correct) ===")
    print(f"[True ] {matrix[0]}")
    print(f"[False] {matrix[1]}")
    print("\n=== Chi-squared Test ===")
    print(f"chi2 = {chi['statistic']:.6f}")
    print(f"p-value = {chi['p_value']:.6f}")
    print(f"Interpretation: {chi['interpretation']}")

    if args.verbose:
        print("\nTop-10: tool_correct=True, span_hit=False")
        for rec in summary["top_tool_correct_span_wrong"]:
            print(f"- id={rec['id']} split={rec['split']} query={rec['query']}")

        print("\nTop-10: tool_correct=False, span_hit=True")
        for rec in summary["top_tool_wrong_span_correct"]:
            print(f"- id={rec['id']} split={rec['split']} query={rec['query']}")

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved dissociation summary to {args.save}")


if __name__ == "__main__":
    main()
