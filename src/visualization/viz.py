"""Visualization utilities for token attributions."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def _normalize(scores: Sequence[float]) -> np.ndarray:
    """Normalize numeric scores to [0, 1].

    Args:
        scores: Sequence of score values.

    Returns:
        Normalized numpy array.

    Raises:
        None.
    """
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr
    min_v = float(arr.min())
    max_v = float(arr.max())
    if abs(max_v - min_v) < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def token_highlight_html(tokens: Sequence[str], scores: Sequence[float]) -> str:
    """Create HTML with attribution-colored token spans.

    Args:
        tokens: Token strings.
        scores: Attribution scores.

    Returns:
        HTML snippet string.

    Raises:
        ValueError: If lengths are inconsistent.
    """
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have same length")

    norm = _normalize(scores)
    cmap = cm.get_cmap("YlOrRd")
    parts = ["<div style='font-family:monospace;line-height:2'>"]

    for token, score in zip(tokens, norm):
        r, g, b, _ = cmap(float(score))
        rgba = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.75)"
        parts.append(
            "<span style='padding:2px 4px;margin:1px;border-radius:4px;"
            f"background:{rgba};'>{escape(token)}</span>"
        )

    parts.append("</div>")
    return "".join(parts)


def plot_token_bar(tokens: Sequence[str], scores: Sequence[float], top_k: int = 15):
    """Plot a horizontal bar chart of top-k token attributions.

    Args:
        tokens: Token strings.
        scores: Attribution scores.
        top_k: Number of top tokens to plot.

    Returns:
        Matplotlib figure and axis.

    Raises:
        ValueError: If lengths are inconsistent.
    """
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have same length")

    pairs = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)[:top_k]
    labels = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.barh(labels, values, color="#cc4c02")
    ax.set_xlabel("Attribution Score")
    ax.set_title(f"Top-{top_k} Token Attribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig, ax


def plot_attention_heatmap(matrix: np.ndarray, tokens: Sequence[str], max_tokens: int = 40):
    """Plot attention rollout heatmap for a token sequence.

    Args:
        matrix: Rollout matrix [seq, seq].
        tokens: Token sequence.
        max_tokens: Maximum displayed token count.

    Returns:
        Matplotlib figure and axis.

    Raises:
        ValueError: If matrix is not square or misaligned with tokens.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    if len(tokens) != arr.shape[0]:
        raise ValueError("token length must match matrix size")

    limit = min(max_tokens, arr.shape[0])
    clipped = arr[:limit, :limit]
    clipped_tokens = list(tokens)[:limit]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    image = ax.imshow(clipped, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(limit))
    ax.set_yticks(range(limit))
    ax.set_xticklabels(clipped_tokens, rotation=90, fontsize=7)
    ax.set_yticklabels(clipped_tokens, fontsize=7)
    ax.set_title("Attention Rollout Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax


def parse_args() -> argparse.Namespace:
    """Parse CLI args for visualization generation."""
    parser = argparse.ArgumentParser(description="Generate AgentXplain visualizations")
    parser.add_argument("--results", type=Path, required=True, help="Attribution results JSON")
    parser.add_argument("--save", type=Path, required=True, help="Output directory")
    return parser.parse_args()


def _sanitize_method(name: str) -> str:
    """Make method names filesystem-safe."""
    return name.replace(" ", "_").replace("/", "_")


def generate_visualizations(results_path: Path, out_dir: Path) -> None:
    """Generate the figure artifacts from a results JSON file."""
    records = json.loads(results_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    method_names = set()

    for idx, rec in enumerate(records, start=1):
        trace_name = f"trace_{idx:03d}"
        attributions = rec.get("attributions", {}) if isinstance(rec, dict) else {}

        # Create highlight HTML from attention_rollout if present, otherwise first method.
        method_for_html = "attention_rollout" if "attention_rollout" in attributions else next(iter(attributions), None)
        if method_for_html:
            ranked = attributions[method_for_html]
            tokens = [tok for tok, _ in ranked]
            scores = [float(score) for _, score in ranked]
            html = token_highlight_html(tokens, scores)
            (out_dir / f"highlight_{trace_name}.html").write_text(html, encoding="utf-8")

        for method, ranked in attributions.items():
            method_names.add(method)
            tokens = [tok for tok, _ in ranked]
            scores = [float(score) for _, score in ranked]

            fig, _ = plot_token_bar(tokens, scores, top_k=min(15, max(1, len(tokens))))
            fig.savefig(out_dir / f"bar_{trace_name}_{_sanitize_method(method)}.png", dpi=200)
            plt.close(fig)

            trig = str(rec.get("planted_trigger", "")).lower().split()
            top10 = [str(tok).lower().strip("▁Ġ.,?!") for tok, _ in ranked[:10]]
            hit = 1.0 if set(trig) & set(top10) else 0.0
            summary_rows.append((trace_name, method, hit))

    # Summary heatmap across methods/traces using trigger-hit in top10.
    method_list = sorted(method_names)
    trace_list = sorted({t for t, _, _ in summary_rows})
    heat = np.zeros((len(trace_list), len(method_list)), dtype=np.float32)
    trace_idx = {t: i for i, t in enumerate(trace_list)}
    method_idx = {m: i for i, m in enumerate(method_list)}
    for t, m, v in summary_rows:
        heat[trace_idx[t], method_idx[m]] = v

    fig, ax = plt.subplots(figsize=(max(6, len(method_list) * 1.2), max(4, len(trace_list) * 0.6)), dpi=200)
    im = ax.imshow(heat, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(method_list)))
    ax.set_xticklabels(method_list, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(trace_list)))
    ax.set_yticklabels(trace_list, fontsize=8)
    ax.set_title("Trigger Hit@10 Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_heatmap.png", dpi=200)
    plt.close(fig)

    print(f"Saved visualization artifacts to {out_dir}")


def main() -> None:
    """CLI entrypoint for generating all figure artifacts."""
    args = parse_args()
    generate_visualizations(args.results, args.save)


if __name__ == "__main__":
    main()
