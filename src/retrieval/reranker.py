import hashlib
import re
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


def _normalize_key_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _hash_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _exact_duplicate_key(doc: dict[str, Any]) -> tuple[str, str]:
    content = _normalize_key_text(doc.get("content"))
    if content:
        return ("content", _hash_key(content))

    return ("doc", _hash_key(_normalize_key_text(_format_doc_for_rerank(doc))))


def _diversity_key(doc: dict[str, Any]) -> tuple[str, str] | None:
    file_name = _normalize_key_text(doc.get("file_name"))
    section_title = _normalize_key_text(doc.get("section_title"))

    if file_name and section_title:
        return (file_name, section_title)

    page_number = doc.get("page_number")
    if file_name and page_number is not None:
        return (file_name, str(page_number))

    title = _normalize_key_text(doc.get("title"))
    if file_name or title:
        return (file_name, title)

    return None


def _remove_exact_duplicates(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique_docs = []

    for doc in docs:
        key = _exact_duplicate_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def _select_diverse_docs(
    docs: list[dict[str, Any]],
    top_k: Optional[int],
) -> list[dict[str, Any]]:
    unique_docs = _remove_exact_duplicates(docs)
    if top_k is None:
        return unique_docs

    selected = []
    deferred = []
    seen_groups = set()

    for doc in unique_docs:
        group_key = _diversity_key(doc)
        if group_key and group_key in seen_groups:
            deferred.append(doc)
            continue

        selected.append(doc)
        if group_key:
            seen_groups.add(group_key)

        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        selected_ids = {id(doc) for doc in selected}
        for doc in deferred:
            if id(doc) in selected_ids:
                continue
            selected.append(doc)
            if len(selected) >= top_k:
                break

    return selected


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

    selected_docs = _select_diverse_docs(reranked_docs, top_k)

    for rank, doc in enumerate(selected_docs, start=1):
        doc["final_rank"] = rank

    return selected_docs
