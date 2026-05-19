from typing import Optional
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text
from src.embeddings.sparse_embed import embed_sparse_text 

def _format_results(results):
    return [{"point_id": r.id, "score": r.score, **r.payload} for r in results.points]

def _is_missing_vector_name_error(error, *vector_names):
    message = str(error)
    return (
        "Not existing vector name" in message
        and any(vector_name in message for vector_name in vector_names)
    )

def _query_dense_points(
    collection_name: str,
    dense_vector,
    top_k: int,
    score_threshold: float,
    query_filter: Optional[models.Filter] = None,
    using_dense: bool = True,
):
    query_kwargs = dict(
        collection_name=collection_name,
        query=dense_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )

    if using_dense:
        query_kwargs["using"] = "dense"

    return client.query_points(**query_kwargs)

def _query_dense_points_with_fallback(
    collection_name: str,
    dense_vector,
    top_k: int,
    score_threshold: float,
    query_filter: Optional[models.Filter] = None,
):
    try:
        return _query_dense_points(
            collection_name,
            dense_vector,
            top_k,
            score_threshold,
            query_filter,
            using_dense=True,
        )
    except UnexpectedResponse as e:
        if not _is_missing_vector_name_error(e, "dense"):
            raise
        return _query_dense_points(
            collection_name,
            dense_vector,
            top_k,
            score_threshold,
            query_filter, 
            using_dense=False,
        )

def retrieve(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.7,
    search_mode: str = "hybrid",
    query_filter: Optional[models.Filter] = None,
    rerank_enabled: bool = False,
    candidate_k: Optional[int] = None,
    rerank_model: Optional[str] = None,
):
    search_top_k = top_k
    if rerank_enabled:
        search_top_k = max(top_k, candidate_k or top_k)

    if search_mode == "vector":
        docs = vector_search(collection_name, embed_provider, query, search_top_k, score_threshold, query_filter)
    elif search_mode == "keyword":
        docs = keyword_search(collection_name, query, search_top_k, query_filter)
    elif search_mode == "hybrid":
        docs = hybrid_search(collection_name, embed_provider, query, search_top_k, score_threshold, query_filter)
    else:
        raise ValueError(f"지원하지 않는 search_mode: '{search_mode}'")

    if not rerank_enabled:
        return docs[:top_k]

    from src.retrieval.reranker import DEFAULT_RERANKER_MODEL, rerank

    return rerank(
        query=query,
        docs=docs,
        top_k=top_k,
        model_name=rerank_model or DEFAULT_RERANKER_MODEL,
    )


def vector_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: float,
    query_filter: Optional[models.Filter] = None,
):
    dense_vector = embed_text(query, provider=embed_provider)
    results = _query_dense_points_with_fallback(
        collection_name,
        dense_vector,
        top_k,
        score_threshold,
        query_filter
    )

    return _format_results(results)

def keyword_search(
    collection_name: str,
    query: str,
    top_k: int,
    query_filter: Optional[models.Filter] = None,
):
    sparse_vector = embed_sparse_text(query)

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=sparse_vector,
            using="sparse",
            limit=top_k,
            query_filter=query_filter
        )
    except UnexpectedResponse as e:
        if _is_missing_vector_name_error(e, "sparse"):
            raise RuntimeError(
                "현재 Qdrant 컬렉션에 sparse 벡터가 없어 keyword 검색을 실행할 수 없습니다. "
                "--search_mode vector를 사용하거나 --parse --ingest로 컬렉션을 다시 생성하세요."
            ) from e
        raise

    return _format_results(results)

def hybrid_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: float = None,
    query_filter: Optional[models.Filter] = None,
):
    dense_vector = embed_text(query, provider=embed_provider)
    sparse_vector = embed_sparse_text(query)

    try:
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_k * 2,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=top_k * 2,
                    filter=query_filter,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            query_filter=query_filter,
        )
    except UnexpectedResponse as e:
        if not _is_missing_vector_name_error(e, "dense", "sparse"):
            raise
        results = _query_dense_points_with_fallback(
            collection_name,
            dense_vector,
            top_k,
            score_threshold,
            query_filter, 
        )

    return _format_results(results)
