"""Gradient x Input saliency for tool-selection logits."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize saliency values to [0, 1].

    Args:
        scores: Raw saliency vector.

    Returns:
        Normalized saliency vector.

    Raises:
        None.
    """
    arr = np.asarray(scores, dtype=np.float32)
    min_v = float(arr.min(initial=0.0))
    max_v = float(arr.max(initial=0.0))
    if abs(max_v - min_v) < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def gradient_x_input_saliency(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    selected_tool: str,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Compute Gradient x Input attribution at decision position.

    Args:
        model: Causal language model.
        tokenizer: Matching tokenizer.
        prompt: Prompt text for routing.
        selected_tool: Selected tool name token target.
        device: Torch device string.

    Returns:
        Tuple of normalized saliency [seq_len] and ranked token-score list.

    Raises:
        ValueError: If selected_tool cannot be tokenized.
    """
    model = model.to(device)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    tool_ids = tokenizer(selected_tool, add_special_tokens=False)["input_ids"]
    if not tool_ids:
        raise ValueError("selected_tool produced empty tokenization")
    target_id = int(tool_ids[0])

    embedding = model.get_input_embeddings()
    embeds = embedding(input_ids).detach().clone().requires_grad_(True)

    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    target_logit = outputs.logits[:, -1, target_id].sum()

    model.zero_grad(set_to_none=True)
    target_logit.backward()

    grads = embeds.grad
    saliency = (grads * embeds).sum(dim=-1).squeeze(0).detach().cpu().numpy()
    saliency = _normalize(np.abs(saliency))

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())
    ranking = sorted(zip(tokens, saliency.tolist()), key=lambda x: x[1], reverse=True)
    return saliency, ranking


def mock_gradient_saliency(
    tokens: Sequence[str],
    keyword_set: Sequence[str],
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Generate deterministic mock Gradient x Input scores for tests.

    Args:
        tokens: Token sequence.
        keyword_set: Keywords to upweight.

    Returns:
        Tuple of normalized saliency and ranked token-score list.

    Raises:
        None.
    """
    key = {k.lower() for k in keyword_set}
    raw = np.asarray([2.0 if t.lower() in key else 0.25 for t in tokens], dtype=np.float32)
    scores = _normalize(raw)
    ranking = sorted(zip(tokens, scores.tolist()), key=lambda x: x[1], reverse=True)
    return scores, ranking
