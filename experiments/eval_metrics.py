"""Multi-seed metrics aggregation for AgentXplain experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import ttest_rel

METHOD_ORDER = [
    "Attn Rollout",
    "Grad x Input",
    "Token SHAP",
    "Contrastive",
    "TF-IDF Baseline",
    "Random",
]

METRIC_KEYS = {
    "Hit@10": "hit10",
    "Span IoU": "span_iou",
    "Sufficiency": "sufficiency",
    "Comprehensiveness": "comprehensiveness",
}

SPLITS = ["standard", "hard", "paraphrase", "negation"]


def _derive_seed_path(results_path: Path, seed: int) -> Path:
    """Derive seed-specific file path from a base results file path.

    Args:
        results_path: Base results path.
        seed: Seed value.

    Returns:
        Seed-specific results file path.

    Raises:
        None.
    """
    if "{seed}" in str(results_path):
        return Path(str(results_path).format(seed=seed))
    stem = results_path.stem
    return results_path.with_name(f"{stem}_seed{seed}{results_path.suffix}")


def _fallback_method_metrics(record: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """Create minimal method metrics when explicit method_metrics is missing.

    Args:
        record: Trace-level result dictionary.

    Returns:
        Method metrics dictionary keyed by method name.

    Raises:
        None.
    """
    trigger_hit = float(record.get("trigger_hit", 0.0))
    iou = float(record.get("span_iou", 0.0))
    suff = float(record.get("confidence_topk", 0.0))
    comp = max(
        0.0,
        float(record.get("full_confidence", 0.0)) - float(record.get("confidence_without_topk", 0.0)),
    )

    random_hit = float(record.get("random_hit10", 0.0))
    random_iou = float(record.get("random_iou", 0.0))
    random_suff = float(record.get("random_sufficiency", 0.0))
    random_comp = float(record.get("random_comprehensiveness", 0.0))

    base = {
        "Contrastive": {
            "hit10": trigger_hit,
            "span_iou": iou,
            "sufficiency": suff,
            "comprehensiveness": comp,
        },
        "Random": {
            "hit10": random_hit,
            "span_iou": random_iou,
            "sufficiency": random_suff,
            "comprehensiveness": random_comp,
        },
    }

    for method in ["Attn Rollout", "Grad x Input", "Token SHAP", "TF-IDF Baseline"]:
        base[method] = {
            "hit10": float(record.get("hit10", 0.0)),
            "span_iou": float(record.get("span_iou", 0.0)),
            "sufficiency": float(record.get("confidence_topk", 0.0)),
            "comprehensiveness": comp,
        }
    return base


def _extract_method_metrics(record: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """Extract per-method metrics from a record.

    Args:
        record: Trace-level result dictionary.

    Returns:
        Method-name keyed metric dictionary.

    Raises:
        None.
    """
    metrics = record.get("method_metrics")
    if isinstance(metrics, dict):
        return metrics
    return _fallback_method_metrics(record)


def _aggregate_seed(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate per-method metrics for one seed.

    Args:
        records: List of trace records for a single seed.

    Returns:
        Seed summary with overall and split-wise method metrics.

    Raises:
        ValueError: If records are empty.
    """
    if not records:
        raise ValueError("records cannot be empty")

    totals: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    split_totals: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        split: defaultdict(lambda: defaultdict(list)) for split in SPLITS
    }

    for rec in records:
        methods = _extract_method_metrics(rec)
        split = str(rec.get("split", "standard"))
        if split not in split_totals:
            split = "standard"

        for method, vals in methods.items():
            for metric_key in METRIC_KEYS.values():
                totals[method][metric_key].append(float(vals.get(metric_key, 0.0)))
                split_totals[split][method][metric_key].append(float(vals.get(metric_key, 0.0)))

    def summarize(source: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        """Summarize mean values from metric lists.

        Args:
            source: Nested metric lists.

        Returns:
            Nested mean values.

        Raises:
            None.
        """
        out: Dict[str, Dict[str, float]] = {}
        for method in METHOD_ORDER:
            if method not in source:
                continue
            out[method] = {}
            for metric_key in METRIC_KEYS.values():
                arr = source[method].get(metric_key, [])
                out[method][metric_key] = float(np.mean(arr)) if arr else 0.0
        return out

    split_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split in SPLITS:
        split_summary[split] = summarize(split_totals[split])

    return {
        "overall": summarize(totals),
        "splits": split_summary,
    }


def evaluate(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Evaluate a single seed result set.

    Args:
        records: Trace-level records for one seed.

    Returns:
        Single-seed summary dictionary with overall and split metrics.

    Raises:
        ValueError: If records are empty.
    """
    return _aggregate_seed(records)


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Compute mean and standard deviation.

    Args:
        values: Numeric sequence.

    Returns:
        Dictionary with mean and std.

    Raises:
        ValueError: If input is empty.
    """
    if not values:
        raise ValueError("values cannot be empty")
    arr = np.asarray(values, dtype=np.float32)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _compute_p_values(seed_summaries: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Compute paired t-test p-values versus random baseline.

    Args:
        seed_summaries: Per-seed aggregated summaries.

    Returns:
        Nested method->metric->p-value dictionary.

    Raises:
        KeyError: If random baseline is missing.
    """
    p_values: Dict[str, Dict[str, float]] = {}
    random_series: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS.values()}

    for summary in seed_summaries:
        overall = summary["overall"]
        random_metrics = overall["Random"]
        for metric_key in METRIC_KEYS.values():
            random_series[metric_key].append(float(random_metrics.get(metric_key, 0.0)))

    for method in METHOD_ORDER:
        if method == "Random":
            continue
        p_values[method] = {}
        for metric_key in METRIC_KEYS.values():
            series = []
            for summary in seed_summaries:
                overall = summary["overall"]
                series.append(float(overall.get(method, {}).get(metric_key, 0.0)))
            stat = ttest_rel(series, random_series[metric_key])
            p_values[method][metric_key] = float(stat.pvalue) if stat.pvalue is not None else 1.0
    return p_values


def _aggregate_across_seeds(seed_summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate means and stds across seed summaries.

    Args:
        seed_summaries: Per-seed summaries.

    Returns:
        Aggregated report dictionary.

    Raises:
        ValueError: If no seed summaries provided.
    """
    if not seed_summaries:
        raise ValueError("seed_summaries cannot be empty")

    overall: Dict[str, Dict[str, Dict[str, float]]] = {}
    split_report: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
        split: {} for split in SPLITS
    }

    for method in METHOD_ORDER:
        overall[method] = {}
        for metric_key in METRIC_KEYS.values():
            vals = [float(s["overall"].get(method, {}).get(metric_key, 0.0)) for s in seed_summaries]
            overall[method][metric_key] = _mean_std(vals)

    for split in SPLITS:
        for method in METHOD_ORDER:
            split_report[split].setdefault(method, {})
            for metric_key in METRIC_KEYS.values():
                vals = [
                    float(s["splits"].get(split, {}).get(method, {}).get(metric_key, 0.0))
                    for s in seed_summaries
                ]
                split_report[split][method][metric_key] = _mean_std(vals)

    p_values = _compute_p_values(seed_summaries)
    return {
        "overall": overall,
        "split_breakdown": split_report,
        "p_values_vs_random": p_values,
    }


def _format_metric(mean_std: Dict[str, float], star: bool = False) -> str:
    """Format metric as mean ± std string.

    Args:
        mean_std: Dictionary with mean and std keys.
        star: Whether to append significance marker.

    Returns:
        Formatted metric string.

    Raises:
        KeyError: If mean/std keys are absent.
    """
    suffix = "*" if star else ""
    return f"{mean_std['mean']:.2f} ± {mean_std['std']:.2f}{suffix}"


def _print_publication_table(report: Dict[str, object]) -> None:
    """Print publication-style summary table.

    Args:
        report: Aggregated report dictionary.

    Returns:
        None.

    Raises:
        None.
    """
    overall = report["overall"]
    p_values = report["p_values_vs_random"]

    print("Method          | Hit@10        | Span IoU      | Sufficiency   | Comprehensiveness")
    print("----------------|---------------|---------------|---------------|------------------")

    for method in METHOD_ORDER:
        stars = {}
        if method != "Random":
            for label, metric_key in METRIC_KEYS.items():
                stars[metric_key] = bool(p_values.get(method, {}).get(metric_key, 1.0) < 0.05)
        else:
            for metric_key in METRIC_KEYS.values():
                stars[metric_key] = False

        row = overall.get(method, {})
        hit = _format_metric(row.get("hit10", {"mean": 0.0, "std": 0.0}), stars["hit10"])
        iou = _format_metric(row.get("span_iou", {"mean": 0.0, "std": 0.0}), stars["span_iou"])
        suff = _format_metric(row.get("sufficiency", {"mean": 0.0, "std": 0.0}), stars["sufficiency"])
        comp = _format_metric(row.get("comprehensiveness", {"mean": 0.0, "std": 0.0}), stars["comprehensiveness"])

        print(f"{method:<15} | {hit:<13} | {iou:<13} | {suff:<13} | {comp}")


def _print_split_breakdown(report: Dict[str, object]) -> None:
    """Print split-wise breakdown table.

    Args:
        report: Aggregated report dictionary.

    Returns:
        None.

    Raises:
        None.
    """
    split_data = report["split_breakdown"]
    for split in SPLITS:
        print(f"\n=== Split: {split} ===")
        print("Method          | Hit@10        | Span IoU      | Sufficiency   | Comprehensiveness")
        print("----------------|---------------|---------------|---------------|------------------")
        for method in METHOD_ORDER:
            row = split_data.get(split, {}).get(method, {})
            hit = _format_metric(row.get("hit10", {"mean": 0.0, "std": 0.0}))
            iou = _format_metric(row.get("span_iou", {"mean": 0.0, "std": 0.0}))
            suff = _format_metric(row.get("sufficiency", {"mean": 0.0, "std": 0.0}))
            comp = _format_metric(row.get("comprehensiveness", {"mean": 0.0, "std": 0.0}))
            print(f"{method:<15} | {hit:<13} | {iou:<13} | {suff:<13} | {comp}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed namespace.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description="Evaluate AgentXplain metrics across multiple seeds")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/output.json"),
        help="Base results file path; seed-specific files are inferred as *_seed{seed}.json",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="List of random seeds",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("results/metrics_summary.json"),
        help="Output summary JSON path",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for multi-seed metrics summary.

    Args:
        None.

    Returns:
        None.

    Raises:
        OSError: If reading or writing files fails.
    """
    args = parse_args()
    seed_summaries: List[Dict[str, object]] = []
    loaded_paths: List[str] = []

    for seed in args.seeds:
        seed_path = _derive_seed_path(args.results, seed)
        records = json.loads(seed_path.read_text(encoding="utf-8"))
        seed_summaries.append(_aggregate_seed(records))
        loaded_paths.append(str(seed_path))

    report = _aggregate_across_seeds(seed_summaries)
    report["seeds"] = list(args.seeds)
    report["source_files"] = loaded_paths

    _print_publication_table(report)
    _print_split_breakdown(report)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved metrics summary to {args.save}")


if __name__ == "__main__":
    main()
