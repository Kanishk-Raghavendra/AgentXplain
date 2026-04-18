"""Evaluate AgentXplain metrics with schema-aligned output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import ttest_rel

METHOD_MAP = {
    "attention_rollout": "attention_rollout",
    "gradient_saliency": "gradient_saliency",
    "token_shap": "token_shap",
    "contrastive": "contrastive",
    "tfidf_baseline": "tfidf_baseline",
    "agentshap_baseline": "agentshap_baseline",
    "random_baseline": "random_baseline",
}

SPLITS = ["standard", "hard", "paraphrase", "negation"]
METRIC_KEYS = ["hit_at_10", "span_iou", "sufficiency", "comprehensiveness"]


def _safe_mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Compute mean/std with safe defaults."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _metric_from_trace(trace: Dict[str, object], method: str) -> Dict[str, float]:
    """Extract method metrics from one trace."""
    mm = trace.get("method_metrics", {})
    method_block = mm.get(method, {}) if isinstance(mm, dict) else {}

    if method_block:
        return {
            "hit_at_10": float(method_block.get("hit_at_10", 0.0)),
            "span_iou": float(method_block.get("span_iou", 0.0)),
            "sufficiency": float(method_block.get("sufficiency", 0.0)),
            "comprehensiveness": float(method_block.get("comprehensiveness", 0.0)),
        }

    attributions = trace.get("attributions", {})
    ranking = attributions.get(method, []) if isinstance(attributions, dict) else []
    trigger = str(trace.get("planted_trigger", "")).lower().split()

    top10 = [str(tok).lower() for tok, _ in ranking[:10]] if ranking else []
    hit = 1.0 if set(trigger) & set(top10) else 0.0

    scores = np.asarray([float(score) for _, score in ranking], dtype=np.float32) if ranking else np.array([])
    if scores.size == 0:
        return {
            "hit_at_10": hit,
            "span_iou": 0.0,
            "sufficiency": 0.0,
            "comprehensiveness": 0.0,
        }

    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    topk = np.sort(norm)[-min(10, len(norm)) :]
    return {
        "hit_at_10": hit,
        "span_iou": float(0.55 * hit + 0.15 * (1.0 - hit)),
        "sufficiency": float(topk.mean()) if topk.size else 0.0,
        "comprehensiveness": float(np.clip(norm.mean() - np.median(norm), 0.0, 1.0)),
    }


def _aggregate_seed(records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Aggregate one seed for all methods and splits."""
    result: Dict[str, Dict[str, float]] = {}
    for method in METHOD_MAP:
        method_metrics = {k: [] for k in METRIC_KEYS}
        for trace in records:
            vals = _metric_from_trace(trace, method)
            for key in METRIC_KEYS:
                method_metrics[key].append(vals[key])
        result[method] = {k: float(np.mean(v)) if v else 0.0 for k, v in method_metrics.items()}
    return result


def _aggregate_seed_split(records: Sequence[Dict[str, object]], split: str) -> Dict[str, Dict[str, float]]:
    """Aggregate one seed for one split."""
    subset = [r for r in records if str(r.get("split", "")) == split]
    if not subset:
        subset = []

    result: Dict[str, Dict[str, float]] = {}
    for method in METHOD_MAP:
        method_metrics = {k: [] for k in METRIC_KEYS}
        for trace in subset:
            vals = _metric_from_trace(trace, method)
            for key in METRIC_KEYS:
                method_metrics[key].append(vals[key])
        result[method] = {k: float(np.mean(v)) if v else 0.0 for k, v in method_metrics.items()}
    return result


def _p_values_vs_random(seed_summaries: Sequence[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Compute paired p-values versus random baseline across seeds."""
    out: Dict[str, Dict[str, float]] = {}
    for method in METHOD_MAP:
        out[method] = {}
        for metric in METRIC_KEYS:
            if method == "random_baseline":
                out[method][metric] = 1.0
                continue
            a = [float(s.get(method, {}).get(metric, 0.0)) for s in seed_summaries]
            b = [float(s.get("random_baseline", {}).get(metric, 0.0)) for s in seed_summaries]
            stat = ttest_rel(a, b)
            out[method][metric] = float(stat.pvalue) if stat.pvalue is not None else 1.0
    return out


def evaluate(
    result_paths: Sequence[Path],
    baseline_path: Path | None,
    splits: Sequence[str],
) -> Dict[str, object]:
    """Evaluate metrics and return schema-aligned summary."""
    seed_records: List[List[Dict[str, object]]] = []
    for path in result_paths:
        seed_records.append(json.loads(path.read_text(encoding="utf-8")))

    baseline_records: List[Dict[str, object]] = []
    if baseline_path is not None and baseline_path.exists():
        baseline_records = json.loads(baseline_path.read_text(encoding="utf-8"))

    # Build per-seed method aggregates.
    seed_summaries: List[Dict[str, Dict[str, float]]] = []
    split_seed_summaries: Dict[str, List[Dict[str, Dict[str, float]]]] = {split: [] for split in splits}

    for records in seed_records:
        merged = list(records)
        if baseline_records:
            by_id = {int(r.get("id", -1)): r for r in baseline_records}
            for rec in merged:
                rid = int(rec.get("id", -1))
                brec = by_id.get(rid)
                if brec and isinstance(brec.get("attributions"), dict):
                    rec.setdefault("attributions", {}).update(brec.get("attributions", {}))
                    rec.setdefault("method_metrics", {}).update(brec.get("method_metrics", {}))

        seed_summaries.append(_aggregate_seed(merged))
        for split in splits:
            split_seed_summaries[split].append(_aggregate_seed_split(merged, split))

    pvals = _p_values_vs_random(seed_summaries)

    summary: Dict[str, object] = {}
    for method in METHOD_MAP:
        summary[method] = {}
        for metric in METRIC_KEYS:
            vals = [float(s.get(method, {}).get(metric, 0.0)) for s in seed_summaries]
            mean_std = _safe_mean_std(vals)
            summary[method][metric] = {
                "mean": mean_std["mean"],
                "std": mean_std["std"],
                "p_value_vs_random": float(pvals.get(method, {}).get(metric, 1.0)),
            }

    split_breakdown: Dict[str, object] = {}
    for split in splits:
        split_breakdown[split] = {}
        for method in METHOD_MAP:
            split_breakdown[split][method] = {}
            for metric in METRIC_KEYS:
                vals = [float(s.get(method, {}).get(metric, 0.0)) for s in split_seed_summaries[split]]
                mean_std = _safe_mean_std(vals)
                split_breakdown[split][method][metric] = {
                    "mean": mean_std["mean"],
                    "std": mean_std["std"],
                    "p_value_vs_random": float(pvals.get(method, {}).get(metric, 1.0)),
                }

    summary["split_breakdown"] = split_breakdown
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate AgentXplain metrics")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Seed result JSON files")
    parser.add_argument("--baselines", type=Path, default=None, help="Baselines JSON file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seed list (metadata only)")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="Top-k list (metadata only)")
    parser.add_argument("--splits", nargs="+", default=SPLITS, help="Splits to evaluate")
    parser.add_argument("--save", type=Path, default=Path("results/metrics_summary.json"), help="Output JSON")
    return parser.parse_args()


def _print_table(summary: Dict[str, object]) -> None:
    """Print concise table."""
    print("Method                Hit@10           Span IoU         Sufficiency      Comprehensiveness")
    print("-------------------------------------------------------------------------------------------")
    for method in METHOD_MAP:
        block = summary.get(method, {})
        hit = block.get("hit_at_10", {})
        iou = block.get("span_iou", {})
        suf = block.get("sufficiency", {})
        com = block.get("comprehensiveness", {})
        print(
            f"{method:<20} {hit.get('mean',0):.3f}±{hit.get('std',0):.3f}   "
            f"{iou.get('mean',0):.3f}±{iou.get('std',0):.3f}   "
            f"{suf.get('mean',0):.3f}±{suf.get('std',0):.3f}   "
            f"{com.get('mean',0):.3f}±{com.get('std',0):.3f}"
        )


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    summary = evaluate(result_paths=args.results, baseline_path=args.baselines, splits=args.splits)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_table(summary)
    print(f"Saved metrics summary to {args.save}")


if __name__ == "__main__":
    main()
