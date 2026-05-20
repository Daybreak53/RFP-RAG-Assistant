import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
import torch
from sentence_transformers import CrossEncoder


FALLBACK_RERANKER_MODEL = "BAAI/bge-reranker-large"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _load_reranker_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config.get("retrieval", {}).get("rerank", {}) or {}


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


RERANKER_CONFIG = _load_reranker_config()
DEFAULT_RERANKER_MODEL = RERANKER_CONFIG.get("model") or FALLBACK_RERANKER_MODEL
DEFAULT_RERANK_MODEL = DEFAULT_RERANKER_MODEL
DEFAULT_RERANKER_MAX_LENGTH = int(RERANKER_CONFIG.get("max_length", 512))
DEFAULT_RERANK_BATCH_SIZE = int(RERANKER_CONFIG.get("batch_size", 16))
DEFAULT_MAX_CONTENT_CHARS = int(RERANKER_CONFIG.get("max_content_chars", 1800))
DEFAULT_RERANK_SCORE_THRESHOLD = _optional_float(
    RERANKER_CONFIG.get("score_threshold", RERANKER_CONFIG.get("rerank_score_threshold"))
)
DEFAULT_DIVERSITY_PER_GROUP = int(RERANKER_CONFIG.get("diversity_per_group", 1))


@lru_cache(maxsize=2)
def get_reranker(model_name: Optional[str] = None) -> CrossEncoder:
    model_name = model_name or DEFAULT_RERANKER_MODEL
    return CrossEncoder(model_name, max_length=DEFAULT_RERANKER_MAX_LENGTH)


def _get_predict_device() -> Optional[str]:
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"

    return None


def _trim_text(value: Any, max_chars: int = DEFAULT_MAX_CONTENT_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    return f"{text[:head_chars]} ... {text[-tail_chars:]}"


def _format_doc_for_rerank(doc: dict[str, Any]) -> str:
    fields = (
        ("content", _trim_text(doc.get("content"))),
        ("section_title", doc.get("section_title")),
        ("title", doc.get("title")),
        ("organization", doc.get("organization")),
        ("budget", doc.get("budget")),
        ("announcement_date", doc.get("announcement_date")),
        ("bid_start", doc.get("bid_start")),
        ("bid_deadline", doc.get("bid_deadline")),
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
    diversity_per_group: int = DEFAULT_DIVERSITY_PER_GROUP,
) -> list[dict[str, Any]]:
    unique_docs = _remove_exact_duplicates(docs)
    if top_k is None:
        return unique_docs

    if diversity_per_group <= 0:
        return unique_docs[:top_k]

    selected = []
    deferred = []
    group_counts = {}

    for doc in unique_docs:
        group_key = _diversity_key(doc)
        if group_key and group_counts.get(group_key, 0) >= diversity_per_group:
            deferred.append(doc)
            continue

        selected.append(doc)
        if group_key:
            group_counts[group_key] = group_counts.get(group_key, 0) + 1

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
    score_threshold: Optional[float] = DEFAULT_RERANK_SCORE_THRESHOLD,
    diversity_per_group: int = DEFAULT_DIVERSITY_PER_GROUP,
) -> list[dict[str, Any]]:
    if not docs:
        return []

    model_name = model_name or DEFAULT_RERANKER_MODEL
    try:
        model = get_reranker(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load reranker model: {model_name}") from exc

    scorable_docs = []
    pairs = []
    for doc in docs:
        doc_text = _format_doc_for_rerank(doc)
        if not doc_text:
            continue
        scorable_docs.append(doc)
        pairs.append((query, doc_text))

    if not pairs:
        return []

    scores = model.predict(
        pairs,
        batch_size=DEFAULT_RERANK_BATCH_SIZE,
        show_progress_bar=False,
        activation_fn=torch.nn.Sigmoid(),
        convert_to_numpy=True,
        device=_get_predict_device(),
    )

    reranked_docs = []
    for original_rank, (doc, score) in enumerate(zip(scorable_docs, scores), start=1):
        updated_doc = dict(doc)
        updated_doc["retrieval_score"] = updated_doc.get("score")
        updated_doc["rerank_score"] = _as_float(score)
        if score_threshold is not None and updated_doc["rerank_score"] < score_threshold:
            continue
        updated_doc["score"] = updated_doc["rerank_score"]
        updated_doc["rerank_original_rank"] = original_rank
        reranked_docs.append(updated_doc)

    reranked_docs.sort(
        key=lambda doc: (doc["rerank_score"], doc.get("retrieval_score") or 0),
        reverse=True,
    )

    for rank, doc in enumerate(reranked_docs, start=1):
        doc["rerank_rank"] = rank

    selected_docs = _select_diverse_docs(
        reranked_docs,
        top_k,
        diversity_per_group=diversity_per_group,
    )

    for rank, doc in enumerate(selected_docs, start=1):
        doc["final_rank"] = rank

    return selected_docs
