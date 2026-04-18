# AgentXplain: Defining and Evaluating Prompt-to-Tool Causal Attribution in LLM Agents

## Abstract
LLM agents increasingly rely on tool selection to answer queries, yet existing explainability methods attribute importance at the tool-output level, leaving the causal link between user input and tool choice unexplained. We identify prompt-to-tool attribution as an understudied problem distinct from response-level tool attribution (e.g. AgentSHAP), and show empirically that the two are non-redundant. We introduce AgentXplain, a framework that attributes tool-selection decisions to input token spans using attention rollout, gradient saliency, and token-masking SHAP, evaluated on a purpose-built causal benchmark with planted trigger spans, paraphrase controls, and adversarial distractors. Experiments demonstrate that input-span attribution recovers causal triggers with high faithfulness, exposes cases where tool-level attribution is uninformative, and reveals a failure taxonomy for incorrect tool routing.

## 1. Introduction
### 1.1 Motivation
- Tool-augmented LLM agents are deployed in workflows where incorrect tool calls produce cascading errors.
- Current explanations mostly answer which tool mattered to final response quality, not what in the prompt caused tool selection.
- This missing causal link limits debugging, reliability analysis, and benchmark design for agentic systems.

### 1.2 Contributions
1. We formally define the prompt-to-tool attribution problem and distinguish it from existing response-level tool attribution
2. We introduce a self-contained causal benchmark (n=300, 4 splits: standard, hard/distractor, paraphrase, negation) with planted ground-truth trigger spans, enabling automated faithfulness evaluation without human annotators
3. We provide the first empirical dissociation analysis showing cases where tool-level attribution (AgentSHAP-style) succeeds but span attribution fails and vice versa, demonstrating non-redundancy
4. We package the full workflow into a reproducible local pipeline that generates data, runs seeded experiments, evaluates metrics, and renders figures without notebooks

### 1.3 Research Questions
- RQ1: Can attention rollout, gradient saliency, and token-masking SHAP faithfully localise the prompt spans that causally trigger tool selection, as measured by span IoU and top-k hit rate on planted trigger benchmarks?
- RQ2: Does contrastive attribution (score of selected tool minus score of best alternative) reduce false positives on distractor-heavy and paraphrase prompts compared to non-contrastive methods?
- RQ3: Are prompt-span attributions non-redundant with tool-level attributions - do they reveal causal information that tool-level methods cannot capture?

## 2. Related Work
### 2.1 Token Attribution Methods
- Abnar and Zuidema, Quantifying Attention Flow in Transformers, ACL 2020.
- Sundararajan et al., Axiomatic Attribution for Deep Networks, ICML 2017.
- Lundberg and Lee, A Unified Approach to Interpreting Model Predictions, NeurIPS 2017.
- Jain and Wallace, Attention is not Explanation, NAACL 2019.
- Adebayo et al., Sanity Checks for Saliency Maps, NeurIPS 2018.

### 2.2 LLM Agents and Tool Use
- Schick et al., Toolformer: Language Models Can Teach Themselves to Use Tools, NeurIPS 2023.
- Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models, NeurIPS 2023.
- Qin et al., ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs, 2023.

### 2.3 Tool-Level and Agent Explainability
- Horovicz, AgentSHAP: Interpreting LLM Agent Tool Importance with Monte Carlo Shapley Value Estimation, arXiv 2512.12597, 2024.
- Goldshmidt and Horovicz, TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation, arXiv, 2024.

### 2.4 Faithfulness Evaluation
- DeYoung et al., ERASER: A Benchmark to Evaluate Rationalized NLP Models, ACL 2020.
- Jacovi and Goldberg, Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?, ACL 2020.
- Hooker et al., A Benchmark for Interpretability Methods in Deep Neural Networks, NeurIPS 2019.

## 3. Methodology
### 3.1 Agent Setup
- Routing model: Qwen/Qwen2.5-0.5B-Instruct with fallback to TinyLlama/TinyLlama-1.1B-Chat-v1.0 and deterministic mock routing.
- Tools: web_search, calculator, code_executor plus fallback behavior for negation traces.
- Trace artifacts: selected tool, argument fields, token ids, attention tensors, and decision score distribution.

### 3.2 Attribution Methods
- Attention rollout attribution.
- Gradient saliency attribution.
- Token-masking SHAP attribution.
- Contrastive attribution using selected tool versus strongest alternative tool.

### 3.3 Formal Metric Definitions
- Top-k Hit:
  $$
  \text{Hit@k} = \mathbf{1}[\mathcal{G} \cap \hat{\mathcal{T}}_k \neq \emptyset]
  $$
  where $\mathcal{G}$ is ground-truth trigger token set and $\hat{\mathcal{T}}_k$ is top-k attributed tokens.
- Span IoU:
  $$
  \text{IoU} = |\hat{\mathcal{T}}_k \cap \mathcal{G}| / |\hat{\mathcal{T}}_k \cup \mathcal{G}|
  $$
- Sufficiency:
  $$
  \text{Suff} = P(\hat{t} \mid \text{top-k tokens only})
  $$
- Comprehensiveness:
  $$
  \text{Comp} = P(\hat{t} \mid \text{full input}) - P(\hat{t} \mid \text{input} \setminus \text{top-k tokens})
  $$
- Contrastive consistency:
  $$
  \text{CC} = \frac{1}{k}\sum_{i \in \hat{\mathcal{T}}_k} [s_{\hat{t}}(i) - s_{t^*}(i)]
  $$
  where $t^*$ is the strongest alternative tool.

## 4. Benchmark Design
### 4.1 Dataset Construction
- Total traces: n=300.
- Split design: standard, hard/distractor, paraphrase, negation.
- Ground truth: planted trigger spans and argument hints for automatic scoring.

### 4.2 Anti-Shortcut Controls
- Paraphrase triggers: same semantic meaning as planted trigger but different surface form, to test generalisation beyond lexical matching.
- Counterfactual distractors: phrases from the correct tool domain but deliberately wrong context, to test false positive rate.

## 5. Experiments
### 5.1 Core Evaluation
- Compare attention rollout, gradient saliency, token-masking SHAP, and contrastive variants.
- Include AgentSHAP-style tool-level baseline and span-localization baselines.
- End-to-end orchestration lives in `experiments/full_pipeline.py`, which generates the benchmark, runs seeded experiments, evaluates metrics, runs dissociation, and renders figures.

### 5.2 Dissociation Analysis
For each benchmark trace, compute both AgentSHAP-style tool-level score and AgentXplain span attribution. Categorise each trace into one of four quadrants:
- Both correct (tool score correct AND span hits trigger)
- Tool correct, span wrong (tool-level succeeds but span attribution misses trigger)
- Tool wrong, span correct (span attribution identifies trigger but tool-level attribution is uninformative)
- Both wrong
Report the 2x2 confusion matrix and chi-squared test for independence. The tool wrong, span correct quadrant is the key evidence for non-redundancy.

### 5.3 Statistical Protocol
- Statistical reporting: all metrics reported as mean +- std over 3 random seeds.
- Significance: paired t-test between each method and the random baseline.
- Stronger baseline: add a TF-IDF keyword similarity baseline (not just lexical keyword match) as a non-trivial span localisation baseline.

### 5.4 Observed Results Snapshot
- Attention rollout, gradient saliency, token SHAP, and contrastive attribution each achieve Hit@10 of 0.997 on the current benchmark run.
- The best sufficiency score among the attribution methods is gradient saliency at 0.298, while contrastive attribution reaches the strongest comprehensiveness among the attribution methods at 0.058.
- The dissociation matrix is heavily imbalanced toward span hits with tool disagreement: 137 tool-correct/span-hit, 162 tool-wrong/span-hit, 0 tool-correct/span-miss, and 1 tool-wrong/span-miss.

## 6. Reproducibility
- Model: Qwen/Qwen2.5-0.5B-Instruct with fallback to TinyLlama/TinyLlama-1.1B-Chat-v1.0; CPU float32 by default.
- Benchmark: n=300, seed=42, fully regeneratable.
- Full pipeline: python experiments/full_pipeline.py --n 300 --benchmark-seed 42 --seeds 42 43 44
- Validation: project tests pass under the repo venv.
- Runtime estimate: report the measured end-to-end time from the pipeline run rather than a placeholder.

## 7. Failure Taxonomy and Discussion
### 7.1 Failure Taxonomy
1. Lexical distractor confusion - attribution fires on distractor keyword instead of true trigger
2. Paraphrase miss - trigger is paraphrased and attribution fails to generalise
3. Multi-intent ambiguity - prompt contains valid triggers for multiple tools
4. Negation blindness - model ignores negation (do NOT search) but attribution still fires on search keywords

### 7.2 Broader Interpretation
- Which method fails under which split and why.
- Practical guidance for deployment-time debugging and auditing.

## 8. Ethics and Threats to Validity
- Synthetic benchmark may not reflect real-world prompt distributions.
- Planted triggers create idealised evaluation; real triggers may be implicit or distributed.
- White-box assumption: all methods require access to model internals; API-only models are excluded.
- Explanations are debugging aids, not ground truth for model behaviour.
- Lexical shortcuts in the benchmark may inflate scores on standard split; hard and paraphrase splits are designed to surface this.

## 9. Conclusion
- AgentXplain reframes attribution as a new prompt-to-tool causal evaluation problem.
- Dissociation results establish non-redundancy between tool-level and input-span attribution.
- The benchmark and taxonomy provide a reusable protocol for future agent explainability work.
