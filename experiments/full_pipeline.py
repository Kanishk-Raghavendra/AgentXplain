"""End-to-end AgentXplain pipeline: benchmark, experiments, metrics, dissociation, and figures."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dissociation_analysis import analyze
from experiments.eval_metrics import SPLITS, evaluate
from experiments.run_benchmark import ALL_METHODS, run_experiment
from src.agent.agent import DEFAULT_ROUTER_MODEL
from src.benchmark.generate import generate_benchmark, save_benchmark, save_split_benchmarks
from src.visualization.viz import generate_visualizations


def _parse_seeds(seed_args: Sequence[int] | None) -> List[int]:
    """Normalize seed input into a stable list."""
    if not seed_args:
        return [42, 43, 44]
    return list(dict.fromkeys(int(seed) for seed in seed_args))


def run_full_pipeline(
    n: int,
    benchmark_seed: int,
    seeds: Sequence[int],
    methods: Sequence[str],
    model_name: str,
    device: str,
    use_mock_router: bool,
    shap_max_evals: int,
    shap_token_limit: int,
    data_dir: Path,
    results_dir: Path,
) -> dict:
    """Run the full project pipeline and persist all major artifacts."""
    seeds_list = _parse_seeds(seeds)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    traces = generate_benchmark(n=n, seed=benchmark_seed)
    save_split_benchmarks(traces, data_dir)
    save_benchmark(traces, data_dir / "benchmark_full.json")

    benchmark_records = [asdict(trace) for trace in traces]
    seed_paths: List[Path] = []
    first_run_records = None

    for seed in seeds_list:
        seed_path = results_dir / f"output_seed{seed}.json"
        records = run_experiment(
            records=benchmark_records,
            methods=methods,
            seed=seed,
            output_path=seed_path,
            model_name=model_name,
            device=device,
            use_mock_router=use_mock_router,
            shap_max_evals=shap_max_evals,
            shap_token_limit=shap_token_limit,
        )
        seed_paths.append(seed_path)
        if first_run_records is None:
            first_run_records = records

    attribution_path = results_dir / "attribution_results.json"
    attribution_path.write_text(json.dumps(first_run_records or [], indent=2), encoding="utf-8")

    metrics_summary = evaluate(result_paths=seed_paths, baseline_path=None, splits=SPLITS)
    metrics_path = results_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    dissociation_summary = analyze(first_run_records or [])
    dissociation_path = results_dir / "dissociation_summary.json"
    dissociation_path.write_text(json.dumps(dissociation_summary, indent=2), encoding="utf-8")

    figures_dir = results_dir / "figures"
    generate_visualizations(attribution_path, figures_dir)

    return {
        "benchmark_path": data_dir / "benchmark_full.json",
        "seed_paths": seed_paths,
        "attribution_path": attribution_path,
        "metrics_path": metrics_path,
        "dissociation_path": dissociation_path,
        "figures_dir": figures_dir,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full pipeline."""
    parser = argparse.ArgumentParser(description="Run the full AgentXplain pipeline")
    parser.add_argument("--n", type=int, default=300, help="Number of synthetic benchmark traces")
    parser.add_argument("--benchmark-seed", type=int, default=42, help="Random seed for benchmark generation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Experiment seeds")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, help="Methods to run")
    parser.add_argument("--model", type=str, default=DEFAULT_ROUTER_MODEL, help="Local model to try first")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--use-mock-router", action="store_true", help="Force deterministic router")
    parser.add_argument("--shap-max-evals", type=int, default=8, help="Max SHAP evaluations per trace")
    parser.add_argument("--shap-token-limit", type=int, default=16, help="Max query tokens used for SHAP")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"), help="Benchmark output directory")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Pipeline output directory")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    outputs = run_full_pipeline(
        n=args.n,
        benchmark_seed=args.benchmark_seed,
        seeds=args.seeds,
        methods=args.methods,
        model_name=args.model,
        device=args.device,
        use_mock_router=bool(args.use_mock_router),
        shap_max_evals=args.shap_max_evals,
        shap_token_limit=args.shap_token_limit,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )

    print(f"Saved benchmark to {outputs['benchmark_path']}")
    print(f"Saved metrics to {outputs['metrics_path']}")
    print(f"Saved dissociation summary to {outputs['dissociation_path']}")
    print(f"Saved figures to {outputs['figures_dir']}")


if __name__ == "__main__":
    main()