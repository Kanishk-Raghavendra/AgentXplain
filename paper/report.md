# AgentXplain Project Report

## Summary
AgentXplain studies prompt-to-tool causal attribution for LLM agents. The cleaned project now runs as a single local pipeline: benchmark generation, seeded attribution experiments, metric aggregation, dissociation analysis, and figure export. The notebooks were removed and replaced by a reproducible CLI workflow.

## Pipeline
The main entry point is `experiments/full_pipeline.py`.

It performs the following steps:

1. Generate the synthetic benchmark with planted trigger spans.
2. Run the attribution experiment across the configured seeds.
3. Write `results/output_seed*.json` and `results/attribution_results.json`.
4. Aggregate metrics into `results/metrics_summary.json`.
5. Run dissociation analysis and write `results/dissociation_summary.json`.
6. Render figures into `results/figures/`.

The default router model is `Qwen/Qwen2.5-0.5B-Instruct`, with fallback to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, then deterministic mock routing.

## Current Results
These numbers are from the latest project outputs in `results/`.

### Attribution quality
- Attention rollout: Hit@10 0.997, Span IoU 0.549, Sufficiency 0.234, Comprehensiveness 0.068
- Gradient saliency: Hit@10 0.997, Span IoU 0.549, Sufficiency 0.298, Comprehensiveness 0.040
- Token SHAP: Hit@10 0.997, Span IoU 0.549, Sufficiency 0.231, Comprehensiveness 0.070
- Contrastive attribution: Hit@10 0.997, Span IoU 0.549, Sufficiency 0.262, Comprehensiveness 0.058
- TF-IDF baseline: Hit@10 0.997, Span IoU 0.549, Sufficiency 0.148, Comprehensiveness 0.082
- AgentSHAP-style baseline: Hit@10 0.797, Span IoU 0.469, Sufficiency 0.617, Comprehensiveness 0.033

### Dissociation analysis
The latest dissociation confusion matrix is:

- tool_correct_span_hit: 137
- tool_correct_span_miss: 0
- tool_wrong_span_hit: 162
- tool_wrong_span_miss: 1

This means the current run is dominated by the non-redundant quadrant where span attribution finds the planted trigger even when tool prediction is wrong. The saved chi-squared statistic is 0.0 with p-value 1.0, so the test is not informative for this particular synthetic matrix.

## Interpretation
The project now supports the core claim that prompt-span attribution and tool-level attribution answer different questions. The strongest practical signal in the current outputs is the large number of tool-wrong/span-hit cases, which is the clearest evidence that span attribution can recover causal prompt evidence even when tool routing fails.

The present benchmark also shows that simple lexical baselines remain strong on the synthetic data, so the hard, paraphrase, and negation splits are essential. The report should therefore emphasize the benchmark design and the dissociation findings rather than only the average scores.

## Verification
The repository test suite passes in the project environment.
