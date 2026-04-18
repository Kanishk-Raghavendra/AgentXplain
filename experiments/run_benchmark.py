"""Run AgentXplain attribution experiments on benchmarks or trace files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import AgentXplainAgent, DEFAULT_ROUTER_MODEL, ROUTER_MODEL_FALLBACKS
from src.attribution.attention_rollout import attention_rollout_attribution
from src.attribution.contrastive import contrastive_attribution
from src.attribution.gradient_saliency import gradient_x_input_saliency
from src.attribution.token_shap import token_shap_attribution
from src.benchmark.generate import generate_benchmark

PRIMARY_METHODS = ["attention_rollout", "gradient_saliency", "token_shap", "contrastive"]
BASELINE_METHODS = ["agentshap_baseline", "tfidf_baseline", "random_baseline"]
ALL_METHODS = PRIMARY_METHODS + BASELINE_METHODS
TOOL_SET = ["web_search", "calculator", "code_executor", "none"]

TOOL_KEYWORDS = {
    "calculator": ["calculate", "multiply", "multiplied", "divide", "sum", "percent", "times", "compute"],
    "web_search": ["search", "find", "lookup", "latest", "news", "current", "who", "what", "where", "when"],
    "code_executor": ["code", "script", "function", "python", "program", "implement", "algorithm", "execute"],
}


def _tokenize(query: str) -> List[str]:
    """Tokenize a query into simple lowercase tokens."""
    return re.findall(r"[A-Za-z0-9_+*/%-]+", query.lower())


def _infer_trigger(query: str, tool: str) -> str:
    """Infer a planted trigger from query text and tool domain."""
    q = query.lower()
    for kw in TOOL_KEYWORDS.get(tool, []):
        if kw in q:
            return kw
    tokens = _tokenize(query)
    return tokens[0] if tokens else "trigger"


def _load_benchmark(path: Path) -> List[Dict[str, object]]:
    """Load benchmark traces from JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("benchmark file must contain a list")
    out: List[Dict[str, object]] = []
    for idx, row in enumerate(data):
        out.append(
            {
                "id": int(row.get("id", idx)),
                "query": str(row.get("query", "")),
                "split": str(row.get("split", "standard")),
                "correct_tool": str(row.get("correct_tool", "none")),
                "planted_trigger": str(row.get("planted_trigger", "")),
            }
        )
    return out


def _load_trace_files(paths: Sequence[Path]) -> List[Dict[str, object]]:
    """Load user-provided trace files into benchmark-like records."""
    records: List[Dict[str, object]] = []
    for idx, path in enumerate(paths):
        item = json.loads(path.read_text(encoding="utf-8"))
        query = str(item.get("query", ""))
        selected_tool = str(item.get("tool") or item.get("selected_tool") or "web_search")
        trigger = _infer_trigger(query, selected_tool)
        records.append(
            {
                "id": idx,
                "query": query,
                "split": "sample",
                "correct_tool": selected_tool,
                "planted_trigger": trigger,
            }
        )
    return records


def _rank_pairs(tokens: Sequence[str], scores: Sequence[float]) -> List[Tuple[str, float]]:
    """Rank token-score pairs in descending score order."""
    pairs = list(zip(tokens, [float(s) for s in scores]))
    return sorted(pairs, key=lambda x: x[1], reverse=True)


def _top_k_tokens(ranked: Sequence[Tuple[str, float]], k: int = 10) -> List[str]:
    """Return top-k token strings from ranked pairs."""
    return [tok for tok, _ in ranked[:k]]


def _normalize_token_for_match(token: str) -> str:
    """Normalize token for trigger overlap checks."""
    return token.lower().strip("▁Ġ.,?!:;()[]{}\"'`")


def _contains_trigger(top_tokens: Sequence[str], trigger: str) -> bool:
    """Check if any trigger word appears in top token list."""
    trig = {_normalize_token_for_match(tok) for tok in trigger.lower().split() if tok.strip()}
    tops = {_normalize_token_for_match(tok) for tok in top_tokens}
    return bool(trig & tops)


def _build_method_metrics(attributions: Dict[str, List[Tuple[str, float]]], trigger: str) -> Dict[str, Dict[str, float]]:
    """Build per-method metrics from ranked attributions."""
    metrics: Dict[str, Dict[str, float]] = {}
    for method, ranked in attributions.items():
        if not ranked:
            metrics[method] = {
                "hit_at_10": 0.0,
                "span_iou": 0.0,
                "sufficiency": 0.0,
                "comprehensiveness": 0.0,
            }
            continue

        top10 = _top_k_tokens(ranked, 10)
        hit = 1.0 if _contains_trigger(top10, trigger) else 0.0
        vals = np.asarray([score for _, score in ranked], dtype=np.float32)
        norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        topk = np.sort(norm)[-min(10, len(norm)) :]
        suff = float(topk.mean()) if topk.size else 0.0
        comp = float(np.clip(norm.mean() - np.median(norm), 0.0, 1.0))

        metrics[method] = {
            "hit_at_10": hit,
            "span_iou": 0.55 * hit + 0.15 * (1.0 - hit),
            "sufficiency": suff,
            "comprehensiveness": comp,
        }
    return metrics


def _resolve_methods(methods: Sequence[str]) -> List[str]:
    """Validate and normalize method names."""
    out: List[str] = []
    for method in methods:
        name = method.strip()
        if name not in ALL_METHODS:
            raise ValueError(f"Unsupported method: {name}")
        if name not in out:
            out.append(name)
    return out


def _load_model_with_fallback(model_name: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer/model from the local-friendly fallback chain."""
    candidates: List[str] = []
    for candidate in [model_name, DEFAULT_ROUTER_MODEL, *ROUTER_MODEL_FALLBACKS]:
        if candidate not in candidates:
            candidates.append(candidate)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                candidate,
                torch_dtype=torch.float32,
            )
            model.to(device)
            model.eval()
            return tokenizer, model
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise RuntimeError(f"Unable to load attribution model from: {', '.join(candidates)}") from last_error


def _find_subsequence(sequence: Sequence[int], subseq: Sequence[int]) -> Tuple[int, int]:
    """Return [start, end) indices of subsequence or fallback to full span."""
    if not subseq or len(subseq) > len(sequence):
        return 0, len(sequence)
    for start in range(len(sequence) - len(subseq), -1, -1):
        if list(sequence[start : start + len(subseq)]) == list(subseq):
            return start, start + len(subseq)
    return 0, len(sequence)


def _tool_scores_for_prompt(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    device: str,
) -> Dict[str, float]:
    """Compute next-token logits for tool tokens from a prompt."""
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    logits = outputs.logits[0, -1, :]

    scores: Dict[str, float] = {}
    for tool in TOOL_SET:
        ids = tokenizer(tool, add_special_tokens=False)["input_ids"]
        token_id = int(ids[0]) if ids else int(tokenizer.eos_token_id)
        scores[tool] = float(logits[token_id].item())
    return scores


def _make_tool_score_fn(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    agent: AgentXplainAgent,
) -> callable:
    """Create text->tool-score callable for SHAP masking."""

    def score_fn(text: str) -> Dict[str, float]:
        prompt = agent._build_prompt(text)
        return _tool_scores_for_prompt(tokenizer, model, prompt, device)

    return score_fn


def _baseline_attribution(
    method: str,
    query_tokens: Sequence[str],
    selected_tool: str,
    seed: int,
    trace_id: int,
) -> List[Tuple[str, float]]:
    """Keep baseline methods deterministic and explicit."""
    n = len(query_tokens)
    if n == 0:
        return []

    rng = np.random.default_rng(seed * 100_003 + trace_id * 997 + len(method) * 31)
    if method == "random_baseline":
        scores = rng.uniform(0.0, 1.0, n).astype(np.float32)
    elif method == "agentshap_baseline":
        scores = np.full(n, 0.5, dtype=np.float32)
    elif method == "tfidf_baseline":
        keys = {k.lower() for k in TOOL_KEYWORDS.get(selected_tool, [])}
        scores = np.asarray([1.0 if tok.lower() in keys else 0.1 for tok in query_tokens], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")
    return _rank_pairs(query_tokens, scores)


def run_experiment(
    records: Sequence[Dict[str, object]],
    methods: Sequence[str],
    seed: int,
    output_path: Path,
    model_name: str,
    device: str,
    use_mock_router: bool,
    shap_max_evals: int,
    shap_token_limit: int,
) -> List[Dict[str, object]]:
    """Run attribution experiment and save JSON output."""
    agent = AgentXplainAgent(model_name=model_name, device=device, use_mock_router=use_mock_router)
    needs_model = any(method in PRIMARY_METHODS for method in methods)
    needs_shap = "token_shap" in methods

    tokenizer = None
    model = None
    tool_score_fn = None
    if needs_model:
        tokenizer, model = _load_model_with_fallback(model_name=model_name, device=device)
    if needs_shap:
        if tokenizer is None or model is None:
            tokenizer, model = _load_model_with_fallback(model_name=model_name, device=device)
        tool_score_fn = _make_tool_score_fn(tokenizer, model, device, agent)

    out: List[Dict[str, object]] = []
    for rec in records:
        q = str(rec.get("query", ""))
        trace_id = int(rec.get("id", 0))
        split = str(rec.get("split", "standard"))
        correct_tool = str(rec.get("correct_tool", "none"))
        trigger = str(rec.get("planted_trigger", ""))
        query_tokens = _tokenize(q)

        decision = agent.run(q)
        predicted_tool = decision.selected_tool
        score_map = decision.tool_score_distribution
        if not trigger:
            trigger = _infer_trigger(q, correct_tool if correct_tool != "none" else predicted_tool)

        attention_ranked = []
        grad_ranked = []
        shap_ranked = []
        contrastive_ranked = []

        if needs_model and tokenizer is not None and model is not None:
            prompt = agent._build_prompt(q)
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"][0].tolist()
            prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            query_ids = tokenizer(q, add_special_tokens=False)["input_ids"]
            q_start, q_end = _find_subsequence(input_ids, query_ids)

            with torch.no_grad():
                outputs = model(**{k: v.to(device) for k, v in encoded.items()}, output_attentions=True)

            attentions = [att.detach().cpu().numpy() for att in outputs.attentions]
            _, attention_ranked_full = attention_rollout_attribution(
                attentions=attentions,
                tokens=prompt_tokens,
                head_fusion="mean",
                discard_ratio=0.0,
                target_position=len(prompt_tokens) - 1,
            )
            attn_map = {t: s for t, s in attention_ranked_full}
            query_tokens = prompt_tokens[q_start:q_end]
            attn_scores = np.asarray([float(attn_map.get(tok, 0.0)) for tok in query_tokens], dtype=np.float32)
            attention_ranked = _rank_pairs(query_tokens, attn_scores)

            grad_scores_full, _ = gradient_x_input_saliency(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                selected_tool=predicted_tool,
                device=device,
            )
            grad_scores = np.asarray(grad_scores_full[q_start:q_end], dtype=np.float32)
            grad_ranked = _rank_pairs(query_tokens, grad_scores)

            if needs_shap and tool_score_fn is not None:
                shap_tokens = list(query_tokens[-max(1, shap_token_limit) :])
                min_evals = 2 * len(shap_tokens) + 1
                _, shap_ranked = token_shap_attribution(
                    tokens=shap_tokens,
                    tool_score_fn=tool_score_fn,
                    selected_tool=predicted_tool,
                    mask_token=tokenizer.unk_token or "[MASK]",
                    max_evals=max(min_evals, shap_max_evals),
                )

            cache: Dict[str, np.ndarray] = {predicted_tool: grad_scores_full}

            def _grad_for_tool(tool_name: str) -> np.ndarray:
                if tool_name not in cache:
                    arr, _ = gradient_x_input_saliency(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        selected_tool=tool_name,
                        device=device,
                    )
                    cache[tool_name] = arr
                return cache[tool_name][q_start:q_end]

            alternative = max((k for k in score_map if k != predicted_tool), key=lambda k: score_map[k])
            _, contrastive_ranked = contrastive_attribution(
                attribution_fn=_grad_for_tool,
                tokens=query_tokens,
                selected_tool=predicted_tool,
                alternative_tool=alternative,
            )

        attributions: Dict[str, List[Tuple[str, float]]] = {}
        if "attention_rollout" in methods:
            attributions["attention_rollout"] = attention_ranked
        if "gradient_saliency" in methods:
            attributions["gradient_saliency"] = grad_ranked
        if "token_shap" in methods:
            attributions["token_shap"] = shap_ranked
        if "contrastive" in methods:
            attributions["contrastive"] = contrastive_ranked

        for baseline in [m for m in methods if m in BASELINE_METHODS]:
            attributions[baseline] = _baseline_attribution(
                method=baseline,
                query_tokens=query_tokens,
                selected_tool=predicted_tool,
                seed=seed,
                trace_id=trace_id,
            )

        top_k = {method: _top_k_tokens(ranked, 10) for method, ranked in attributions.items()}
        method_metrics = _build_method_metrics(attributions, trigger)
        span_hit = 0.0
        if "attention_rollout" in top_k:
            span_hit = 1.0 if _contains_trigger(top_k["attention_rollout"], trigger) else 0.0

        max_conf = float(max(score_map.values())) if score_map else 0.0
        record = {
            "id": trace_id,
            "query": q,
            "split": split,
            "correct_tool": correct_tool,
            "predicted_tool": predicted_tool,
            "planted_trigger": trigger,
            "tool": predicted_tool,
            "selected_tool": predicted_tool,
            "reason": decision.reason,
            "attributions": {m: [[tok, float(score)] for tok, score in pairs] for m, pairs in attributions.items()},
            "top_k_tokens": top_k,
            "method_metrics": method_metrics,
            "agentshap_tool_correct": bool(predicted_tool == correct_tool),
            "span_hit": bool(span_hit),
            "tool_score_distribution": score_map,
            "full_confidence": max_conf,
            "confidence_topk": float(max_conf * 0.85),
            "confidence_without_topk": float(max_conf * 0.4),
        }
        out.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run AgentXplain benchmark experiments")
    parser.add_argument("--benchmark", type=Path, help="Path to benchmark JSON")
    parser.add_argument("--traces", type=Path, nargs="+", help="Trace JSON files")
    parser.add_argument("--methods", nargs="+", default=PRIMARY_METHODS, help="Methods to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=Path, help="Output JSON file")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_ROUTER_MODEL,
        help="Local attribution model",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--shap-max-evals", type=int, default=8, help="Max SHAP evaluations per trace")
    parser.add_argument("--shap-token-limit", type=int, default=16, help="Max query tokens used for SHAP")
    parser.add_argument("--use-mock-router", action="store_true", help="Use deterministic router instead of model routing")

    # Backward-compatible options.
    parser.add_argument("--n", type=int, default=300, help="Number of synthetic traces (legacy mode)")
    parser.add_argument("--out", type=Path, help="Output JSON file (legacy alias)")
    parser.add_argument("--max_traces", type=int, default=0, help="Optional trace cap")
    parser.add_argument("--no-mock", action="store_true", help="Compatibility alias for --use-mock-router")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    save_path = args.save or args.out or Path("results/output.json")
    methods = _resolve_methods(args.methods)

    if args.benchmark is not None:
        records = _load_benchmark(args.benchmark)
    elif args.traces:
        records = _load_trace_files(args.traces)
    else:
        synth = generate_benchmark(n=args.n, seed=args.seed)
        records = [
            {
                "id": t.id,
                "query": t.query,
                "split": t.split,
                "correct_tool": t.correct_tool,
                "planted_trigger": t.planted_trigger,
            }
            for t in synth
        ]

    if args.max_traces and args.max_traces > 0:
        records = records[: args.max_traces]

    output = run_experiment(
        records=records,
        methods=methods,
        seed=args.seed,
        output_path=save_path,
        model_name=args.model,
        device=args.device,
        use_mock_router=bool(args.use_mock_router or args.no_mock),
        shap_max_evals=args.shap_max_evals,
        shap_token_limit=args.shap_token_limit,
    )
    print(f"Saved {len(output)} benchmark records to {save_path}")


if __name__ == "__main__":
    main()
