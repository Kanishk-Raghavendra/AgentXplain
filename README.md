# AgentXplain
Prompt-to-Tool Causal Attribution for LLM Agents
Team 27 - Agentic XAI | PES University

![Build Passing](https://img.shields.io/badge/build-passing-brightgreen)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![arXiv](https://img.shields.io/badge/arXiv-placeholder-b31b1b)

AgentXplain addresses a gap left by AgentSHAP (Horovicz, 2024) and similar tool-level attribution methods: while they explain which tool contributed to the final response, they cannot explain which tokens in the user's prompt caused the agent to select that tool in the first place. We show empirically that these two attribution levels are non-redundant.

## Team

| Name | SRN | Role |
|------|-----|------|
| Balasa Chenchu Purandar Puneet | PES1UG23M906 | Agent setup & tool traces |
| Mahendra Kausik V | PES1UG23AM163 | Attribution methods |
| Kanishk Raghavendra | PES1UG23AM135 | Benchmark construction & faithfulness |
| Ananthesha MS | PES1UG23AM161 | Dissociation analysis, baselines & paper |

## Pipeline

```text
User prompt (tokens)
       |
       v
+---------------------+
|   LLM Agent         |  --> Tool selected: [search / calc / code]
|   (Mistral-7B)      |  --> Args, Reason
+---------------------+
       | attention weights
       | token gradients
       v
+-----------------------------------------+
|         Attribution Layer               |
|  +---------+ +----------+ +---------+   |
|  | Attn    | | GradxIn  | | Token   |   |
|  | Rollout | | Saliency | |  SHAP   |   |
|  +---------+ +----------+ +---------+   |
|         + Contrastive extension         |
+-----------------------------------------+
       |
       v
+-----------------------------------------+
|      Faithfulness Evaluation            |
|  Sufficiency | Comprehensiveness        |
|  Span IoU    | Top-k Hit Rate           |
|  Dissociation Analysis vs AgentSHAP     |
+-----------------------------------------+
       |
       v
Failure Taxonomy + Publication-ready results table
```

## Quickstart

1. Create a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate benchmark splits and full set:

```bash
python -m src.benchmark.generate --n 300 --seed 42 --out-dir data/synthetic --out data/synthetic/benchmark_full.json
```

4. Run benchmark trace generation:

```bash
python experiments/run_benchmark.py --n 300 --seed 42 --out results/output_seed42.json
```

5. Aggregate metrics across seeds:

```bash
python experiments/eval_metrics.py --results results/output.json --seeds 42 43 44 --save results/metrics_summary.json
```

6. Run dissociation analysis:

```bash
python experiments/dissociation_analysis.py --results results/output_seed42.json --save results/dissociation_summary.json --verbose
```

## Benchmark Splits

| Split | n | Purpose |
|-------|---|---------|
| Standard | 150 | Main evaluation |
| Hard (distractor) | 50 | Tests specificity vs cross-domain keywords |
| Paraphrase | 50 | Tests generalisation beyond surface tokens |
| Negation | 50 | Tests causal sensitivity to instruction inversion |

## Methods and Baselines

- Attribution methods: Attention Rollout, Gradient x Input, Token SHAP, Contrastive Attribution.
- Faithfulness metrics: Hit@k, Span IoU, Sufficiency, Comprehensiveness, Contrastive Consistency.
- Baselines: Random, lexical keyword matching, TF-IDF keyword similarity, embedding similarity, AgentSHAP-style uniform span baseline.

## Results Table Placeholder

| Method | Hit@10 | Span IoU | Sufficiency | Comprehensiveness |
|--------|--------|----------|-------------|-------------------|
| Attn Rollout | TBD | TBD | TBD | TBD |
| Grad x Input | TBD | TBD | TBD | TBD |
| Token SHAP | TBD | TBD | TBD | TBD |
| Contrastive | TBD | TBD | TBD | TBD |
| TF-IDF Baseline | TBD | TBD | TBD | TBD |
| Embedding Baseline | TBD | TBD | TBD | TBD |
| Random | TBD | TBD | TBD | TBD |

## Repository Layout

```text
AgentXplain/
├── src/
│   ├── agent/
│   ├── attribution/
│   ├── benchmark/
│   ├── faithfulness/
│   └── visualization/
├── experiments/
├── tests/
├── notebooks/
├── data/synthetic/
├── results/
├── paper/
├── requirements.txt
└── README.md
```

## References

1. Abnar and Zuidema, Quantifying Attention Flow in Transformers, ACL, 2020.
2. Sundararajan et al., Axiomatic Attribution for Deep Networks, ICML, 2017.
3. Lundberg and Lee, A Unified Approach to Interpreting Model Predictions, NeurIPS, 2017.
4. Jain and Wallace, Attention is not Explanation, NAACL, 2019.
5. Adebayo et al., Sanity Checks for Saliency Maps, NeurIPS, 2018.
6. Schick et al., Toolformer: Language Models Can Teach Themselves to Use Tools, NeurIPS, 2023.
7. Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models, NeurIPS, 2023.
8. Qin et al., ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs, 2023.
9. Horovicz, AgentSHAP: Interpreting LLM Agent Tool Importance with Monte Carlo Shapley Value Estimation, arXiv 2512.12597, 2024.
10. Goldshmidt and Horovicz, TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation, arXiv, 2024.
11. DeYoung et al., ERASER: A Benchmark to Evaluate Rationalized NLP Models, ACL, 2020.
12. Jacovi and Goldberg, Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?, ACL, 2020.
13. Hooker et al., A Benchmark for Interpretability Methods in Deep Neural Networks, NeurIPS, 2019.

## License

MIT
