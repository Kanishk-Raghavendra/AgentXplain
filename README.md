# AgentXplain

Prompt-to-tool causal attribution for LLM agents.

AgentXplain explains which prompt tokens caused a tool-routing decision (calculator, web search, or code execution), then evaluates explanation quality with faithfulness metrics and dissociation analysis.

## What this project contains

- A routing agent that outputs tool, args, and reason traces
- Synthetic benchmark generation with `standard`, `hard`, `paraphrase`, and `negation` splits
- Attribution methods:
       - Attention rollout
       - Gradient saliency
       - Token SHAP
       - Contrastive attribution
- Baselines and metrics aggregation across random seeds
- Dissociation analysis and visualization utilities

## Project structure

```text
AgentXplain/
├── data/
│   ├── synthetic/
│   └── traces/
├── experiments/
│   ├── baselines.py
│   ├── dissociation_analysis.py
│   ├── eval_metrics.py
│   ├── full_pipeline.py
│   └── run_benchmark.py
├── paper/
├── results/
├── src/
│   ├── agent/
│   ├── attribution/
│   ├── benchmark/
│   ├── faithfulness/
│   └── visualization/
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Use a single local environment at `.venv/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate quickly

```bash
source .venv/bin/activate
pytest tests/ -v --tb=short
```

## Full pipeline

Run the complete project workflow from benchmark generation through evaluation and figures:

```bash
python experiments/full_pipeline.py --n 300 --benchmark-seed 42 --seeds 42 43 44
```

This produces:

- `data/synthetic/benchmark_full.json`
- `data/synthetic/benchmark_standard.json`
- `data/synthetic/benchmark_hard.json`
- `data/synthetic/benchmark_paraphrase.json`
- `data/synthetic/benchmark_negation.json`
- `results/output_seed42.json`
- `results/output_seed43.json`
- `results/output_seed44.json`
- `results/attribution_results.json`
- `results/metrics_summary.json`
- `results/dissociation_summary.json`
- `results/figures/`

The router and attribution stages use the local free default model `Qwen/Qwen2.5-0.5B-Instruct`, with fallback to `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and then deterministic mock routing.

## Paper and report

- Outline: [paper/outline.md](paper/outline.md)
- Report: [paper/report.md](paper/report.md)

## Docker

Build the container:

```bash
docker build -t agentxplain .
```

Run the test suite inside the container:

```bash
docker run --rm agentxplain
```

Run the router manually:

```bash
docker run --rm agentxplain python src/agent/agent.py --query "What is 145 multiplied by 37?" --save results/trace_docker.json
```

## Notes

- Keep generated artifacts in `results/` and synthetic datasets in `data/synthetic/`.
- Avoid creating multiple virtual environments in the repo root.
- If you regenerate benchmark data, existing files in `data/synthetic/` are overwritten.

## License

MIT
