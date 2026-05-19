from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from sentence_transformers import CrossEncoder


FALLBACK_RERANKER_MODEL = "BAAI/bge-reranker-large"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _load_default_reranker_model() -> str:
    if not CONFIG_PATH.exists():
        return FALLBACK_RERANKER_MODEL

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return (
        config.get("retrieval", {})
        .get("rerank", {})
        .get("model")
        or FALLBACK_RERANKER_MODEL
    )


DEFAULT_RERANKER_MODEL = _load_default_reranker_model()
DEFAULT_RERANK_MODEL = DEFAULT_RERANKER_MODEL


@lru_cache(maxsize=2)
def get_reranker(model_name: Optional[str] = None) -> CrossEncoder:
    model_name = model_name or DEFAULT_RERANKER_MODEL
    return CrossEncoder(model_name)


def _format_doc_for_rerank(doc: dict[str, Any]) -> str:
    fields = (
        ("title", doc.get("title")),
        ("organization", doc.get("organization")),
        ("budget", doc.get("budget")),
        ("announcement_date", doc.get("announcement_date")),
        ("bid_start", doc.get("bid_start")),
        ("bid_deadline", doc.get("bid_deadline")),
        ("section_title", doc.get("section_title")),
        ("content", doc.get("content")),
    )

    return "\n".join(
        f"{name}: {value}"
        for name, value in fields
        if value is not None and str(value).strip()
    )


def _as_float(score: Any) -> float:
    if hasattr(score, "tolist"):
        return _as_float(score.tolist())

    if isinstance(score, (list, tuple)):
        if not score:
            return 0.0
        return _as_float(score[-1])

    return float(score)


def rerank(
    query: str,
    docs: list[dict[str, Any]],
    top_k: Optional[int] = None,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    if not docs:
        return []

    model_name = model_name or DEFAULT_RERANKER_MODEL
    try:
        model = get_reranker(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load reranker model: {model_name}") from exc

    pairs = [(query, _format_doc_for_rerank(doc)) for doc in docs]
    scores = model.predict(pairs)

    reranked_docs = []
    for original_rank, (doc, score) in enumerate(zip(docs, scores), start=1):
        updated_doc = dict(doc)
        updated_doc["retrieval_score"] = updated_doc.get("score")
        updated_doc["rerank_score"] = _as_float(score)
        updated_doc["score"] = updated_doc["rerank_score"]
        updated_doc["rerank_original_rank"] = original_rank
        reranked_docs.append(updated_doc)

    reranked_docs.sort(key=lambda doc: doc["rerank_score"], reverse=True)

    for rank, doc in enumerate(reranked_docs, start=1):
        doc["rerank_rank"] = rank

    if top_k is None:
        return reranked_docs

    return reranked_docs[:top_k]
