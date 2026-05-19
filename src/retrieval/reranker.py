from functools import lru_cache
from typing import Any, Optional

from sentence_transformers import CrossEncoder


DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=2)
def get_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoder:
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
    model_name: str = DEFAULT_RERANK_MODEL,
) -> list[dict[str, Any]]:
    if not docs:
        return []

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
