"""Baseline attribution methods for AgentXplain experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

TOOL_DOMAIN_KEYWORDS = {
    "calculator": {"calculate", "multiply", "divide", "sum", "percent", "times", "plus", "minus"},
    "web_search": {"search", "find", "who", "what", "latest", "news", "current"},
    "code_executor": {"code", "script", "function", "program", "write", "implement", "algorithm"},
}

TOOL_DESCRIPTIONS = {
    "calculator": "A numeric computation tool for arithmetic expressions and equation solving.",
    "web_search": "An information retrieval tool for factual lookup and current updates.",
    "code_executor": "A code execution tool that runs Python snippets and returns outputs.",
}


def _normalize(scores: Sequence[float]) -> np.ndarray:
    """Normalize scores into [0, 1] range.

    Args:
        scores: Raw numeric scores.

    Returns:
        Normalized score array.

    Raises:
        None.
    """
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr
    low = float(arr.min())
    high = float(arr.max())
    if abs(high - low) < 1e-12:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def _tokenize(query: str) -> List[str]:
    """Tokenize a query for baseline scoring.

    Args:
        query: Input query string.

    Returns:
        Lowercased tokens.

    Raises:
        None.
    """
    return query.lower().replace(",", " ").replace(".", " ").split()


def random_baseline(tokens: Sequence[str], seed: int = 42) -> np.ndarray:
    """Return uniform random token scores.

    Args:
        tokens: Input token sequence.
        seed: Random seed.

    Returns:
        Random score vector in [0, 1].

    Raises:
        None.
    """
    rng = np.random.default_rng(seed)
    return rng.random(len(tokens), dtype=np.float32)


def attention_only_baseline(attentions: np.ndarray) -> np.ndarray:
    """Compute non-rollout baseline from last-layer mean attention.

    Args:
        attentions: Last-layer attention [batch, heads, seq, seq].

    Returns:
        Mean attention scores [seq].

    Raises:
        ValueError: If attention shape is invalid.
    """
    arr = np.asarray(attentions, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError("attentions must have shape [batch, heads, seq, seq]")
    return arr[0].mean(axis=0)[-1]


def lexical_keyword_baseline(tokens: Sequence[str], tool_keywords_dict: Dict[str, Sequence[str]]) -> np.ndarray:
    """Score tokens by dictionary keyword overlap.

    Args:
        tokens: Input token sequence.
        tool_keywords_dict: Mapping tool->keywords.

    Returns:
        Lexical baseline scores [seq].

    Raises:
        None.
    """
    keyword_set = {kw.lower() for kws in tool_keywords_dict.values() for kw in kws}
    return np.asarray([1.0 if token.lower() in keyword_set else 0.1 for token in tokens], dtype=np.float32)


def agentshap_baseline(trace: Dict) -> np.ndarray:
    """Convert tool-level attribution into uniform token span scores.

    Args:
        trace: Trace dictionary containing input_token_ids and tool_score_distribution.

    Returns:
        Uniform score array showing inability to localize token spans.

    Raises:
        ValueError: If token ids are missing.
    """
    token_ids = trace.get("input_token_ids", [])
    if not token_ids:
        raise ValueError("trace must include input_token_ids")

    score_map = trace.get("tool_score_distribution", {})
    value = float(np.mean(list(score_map.values()))) if score_map else 0.5
    return np.full(len(token_ids), value, dtype=np.float32)


def tfidf_keyword_baseline(trace: Dict[str, object]) -> List[Tuple[str, float]]:
    """Score tokens with TF-IDF weighted by tool-domain keyword overlap.

    Args:
        trace: Trace dictionary with query, selected_tool, and optional corpus_queries.

    Returns:
        Ranked list of (token, score) tuples.

    Raises:
        ValueError: If query is missing.
    """
    query = str(trace.get("query", "")).strip()
    if not query:
        raise ValueError("trace must include query")

    selected_tool = str(trace.get("selected_tool", trace.get("correct_tool", "web_search")))
    keywords = TOOL_DOMAIN_KEYWORDS.get(selected_tool, set())

    corpus = trace.get("corpus_queries")
    if not isinstance(corpus, list) or not corpus:
        corpus = [query]

    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform([str(x) for x in corpus])
    vocab = vectorizer.vocabulary_

    tokens = _tokenize(query)
    doc_vec = matrix[0]
    scores: List[float] = []
    for token in tokens:
        idx = vocab.get(token)
        tfidf = float(doc_vec[0, idx]) if idx is not None else 0.0
        kw_weight = 1.0 if token in keywords else 0.3
        scores.append(tfidf * kw_weight)

    norm = _normalize(scores)
    ranking = sorted(zip(tokens, norm.tolist()), key=lambda x: x[1], reverse=True)
    return ranking


def embedding_similarity_baseline(trace: Dict[str, object]) -> List[Tuple[str, float]]:
    """Score tokens by embedding similarity to selected tool description.

    Args:
        trace: Trace dictionary with query and selected_tool/correct_tool.

    Returns:
        Ranked list of (token, score) tuples.

    Raises:
        ImportError: If sentence-transformers is not installed.
        ValueError: If query is missing.
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required for embedding_similarity_baseline")

    query = str(trace.get("query", "")).strip()
    if not query:
        raise ValueError("trace must include query")

    selected_tool = str(trace.get("selected_tool", trace.get("correct_tool", "web_search")))
    description = TOOL_DESCRIPTIONS.get(selected_tool, TOOL_DESCRIPTIONS["web_search"])

    tokens = _tokenize(query)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    token_emb = model.encode(tokens)
    tool_emb = model.encode([description])[0]

    tool_norm = float(np.linalg.norm(tool_emb)) + 1e-12
    scores: List[float] = []
    for emb in token_emb:
        denom = (float(np.linalg.norm(emb)) + 1e-12) * tool_norm
        sim = float(np.dot(emb, tool_emb) / denom)
        scores.append(sim)

    norm = _normalize(scores)
    ranking = sorted(zip(tokens, norm.tolist()), key=lambda x: x[1], reverse=True)
    return ranking
