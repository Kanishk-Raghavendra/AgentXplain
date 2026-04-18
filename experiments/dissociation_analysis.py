"""Dissociation analysis for tool-level vs span-level attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from scipy.stats import chi2_contingency


def _trigger_hit_attention(trace: Dict[str, object]) -> bool:
    """Check whether trigger words appear in top-10 attention rollout tokens."""
    trigger = str(trace.get("planted_trigger", "")).lower().split()
    attrs = trace.get("attributions", {})
    ranking = attrs.get("attention_rollout", []) if isinstance(attrs, dict) else []
    top10 = [str(tok).lower().strip("▁Ġ.,?!") for tok, _ in ranking[:10]]
    return bool(set(trigger) & set(top10))


def analyze(records: List[Dict[str, object]]) -> Dict[str, object]:
    """Compute dissociation matrix and statistical test."""
    tt = tf = ft = ff = 0
    examples_span_wins: List[Dict[str, object]] = []
    examples_tool_wins: List[Dict[str, object]] = []

    for rec in records:
        tool_correct = bool(rec.get("predicted_tool") == rec.get("correct_tool"))
        span_hit = _trigger_hit_attention(rec)

        if tool_correct and span_hit:
            tt += 1
        elif tool_correct and not span_hit:
            tf += 1
            examples_tool_wins.append(rec)
        elif (not tool_correct) and span_hit:
            ft += 1
            examples_span_wins.append(rec)
        else:
            ff += 1

    matrix = [[tt, tf], [ft, ff]]
    chi2, pval, _, _ = chi2_contingency(matrix) if sum(sum(r) for r in matrix) > 0 else (0.0, 1.0, None, None)

    def _compact(items: List[Dict[str, object]], n: int = 5) -> List[Dict[str, object]]:
        out = []
        for rec in items[:n]:
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

    return {
        "confusion_matrix": {
            "tool_correct_span_hit": int(tt),
            "tool_correct_span_miss": int(tf),
            "tool_wrong_span_hit": int(ft),
            "tool_wrong_span_miss": int(ff),
        },
        "chi2_statistic": float(chi2),
        "p_value": float(pval),
        "key_examples": {
            "span_correct_tool_wrong": _compact(examples_span_wins),
            "tool_correct_span_wrong": _compact(examples_tool_wins),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run dissociation analysis")
    parser.add_argument("--results", type=Path, required=True, help="Primary results JSON")
    parser.add_argument("--baselines", type=Path, default=None, help="Baselines JSON (optional)")
    parser.add_argument("--save", type=Path, default=Path("results/dissociation_summary.json"), help="Output JSON")
    parser.add_argument("--verbose", action="store_true", help="Print details")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    records = json.loads(args.results.read_text(encoding="utf-8"))
    summary = analyze(records)

    cm = summary["confusion_matrix"]
    print("2x2 dissociation confusion matrix:")
    print(f"  tool_correct_span_hit:  {cm['tool_correct_span_hit']}")
    print(f"  tool_correct_span_miss: {cm['tool_correct_span_miss']}")
    print(f"  tool_wrong_span_hit:    {cm['tool_wrong_span_hit']}")
    print(f"  tool_wrong_span_miss:   {cm['tool_wrong_span_miss']}")
    print(f"Chi-squared: {summary['chi2_statistic']:.4f}")
    print(f"p-value: {summary['p_value']:.6f}")

    if args.verbose and cm["tool_wrong_span_hit"] == 0:
        print("\nNo tool_wrong_span_hit cases found. Showing closest candidates (wrong tool):")
        wrong_tool = [r for r in records if r.get("predicted_tool") != r.get("correct_tool")]
        ranked = sorted(
            wrong_tool,
            key=lambda r: max([float(s) for _, s in r.get("attributions", {}).get("attention_rollout", [])[:10]] or [0.0]),
            reverse=True,
        )
        for rec in ranked[:10]:
            print(f"  id={rec.get('id')} split={rec.get('split')} query={str(rec.get('query',''))[:120]}")

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved dissociation summary to {args.save}")


if __name__ == "__main__":
    main()
