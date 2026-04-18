"""Tests for the full AgentXplain pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from experiments import full_pipeline
from src.benchmark.generate import BenchmarkTrace


def test_run_full_pipeline_writes_expected_artifacts(tmp_path, monkeypatch) -> None:
    """Pipeline should stitch together benchmark, experiment, metrics, dissociation, and figures."""
    trace = BenchmarkTrace(
        id=0,
        query="please calculate 2+2",
        planted_trigger="calculate",
        correct_tool="calculator",
        distractor_keywords=["search", "python"],
        ground_truth_arg_hint="2+2",
        split="standard",
        paraphrase_of=None,
    )

    calls = {"run": [], "viz": []}

    monkeypatch.setattr(full_pipeline, "generate_benchmark", lambda n, seed: [trace])

    def fake_save_split_benchmarks(traces, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    def fake_save_benchmark(traces, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([trace.__dict__ for trace in traces], indent=2), encoding="utf-8")

    monkeypatch.setattr(full_pipeline, "save_split_benchmarks", fake_save_split_benchmarks)
    monkeypatch.setattr(full_pipeline, "save_benchmark", fake_save_benchmark)
    monkeypatch.setattr(full_pipeline, "evaluate", lambda result_paths, baseline_path, splits: {"metrics": True})
    monkeypatch.setattr(full_pipeline, "analyze", lambda records: {"dissociation": True})

    def fake_run_experiment(**kwargs):
        calls["run"].append(kwargs)
        payload = [{"id": 0, "query": "please calculate 2+2", "attributions": {"attention_rollout": [["calculate", 1.0]]}, "planted_trigger": "calculate", "predicted_tool": "calculator", "correct_tool": "calculator"}]
        kwargs["output_path"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def fake_generate_visualizations(results_path: Path, out_dir: Path) -> None:
        calls["viz"].append((results_path, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary_heatmap.png").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(full_pipeline, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(full_pipeline, "generate_visualizations", fake_generate_visualizations)

    outputs = full_pipeline.run_full_pipeline(
        n=1,
        benchmark_seed=7,
        seeds=[11, 13],
        methods=["attention_rollout", "agentshap_baseline"],
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        use_mock_router=True,
        shap_max_evals=8,
        shap_token_limit=8,
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
    )

    assert outputs["benchmark_path"].exists()
    assert outputs["metrics_path"].exists()
    assert outputs["dissociation_path"].exists()
    assert outputs["figures_dir"].exists()
    assert len(calls["run"]) == 2
    assert len(calls["viz"]) == 1
    assert (tmp_path / "results" / "attribution_results.json").exists()
