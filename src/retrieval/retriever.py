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

def retrieve(collection_name, embed_provider, query, top_k=3, score_threshold=0.7, search_mode="hybrid"):
    if search_mode == "vector":
        return vector_search(collection_name, embed_provider, query, top_k, score_threshold)
    elif search_mode == "keyword":
        return keyword_search(collection_name, query, top_k)
    elif search_mode == "hybrid":
        return hybrid_search(collection_name, embed_provider, query, top_k, score_threshold)
    else:
        raise ValueError(f"지원하지 않는 search_mode: '{search_mode}'")

def _query_dense_points(collection_name, dense_vector, top_k, score_threshold, using_dense=True):
    query_kwargs = dict(
        collection_name=collection_name,
        query=dense_vector,
        limit=top_k,
        score_threshold=score_threshold,
    )

    if using_dense:
        query_kwargs["using"] = "dense"

    return client.query_points(**query_kwargs)

def _query_dense_points_with_fallback(collection_name, dense_vector, top_k, score_threshold):
    try:
        return _query_dense_points(
            collection_name,
            dense_vector,
            top_k,
            score_threshold,
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
            using_dense=False,
        )

def vector_search(collection_name, embed_provider, query, top_k, score_threshold):
    dense_vector = embed_text(query, provider=embed_provider)
    results = _query_dense_points_with_fallback(
        collection_name,
        dense_vector,
        top_k,
        score_threshold,
    )

    return _format_results(results)

def keyword_search(collection_name, query, top_k):
    sparse_vector = embed_sparse_text(query)

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=sparse_vector,
            using="sparse",
            limit=top_k
        )
    except UnexpectedResponse as e:
        if _is_missing_vector_name_error(e, "sparse"):
            raise RuntimeError(
                "현재 Qdrant 컬렉션에 sparse 벡터가 없어 keyword 검색을 실행할 수 없습니다. "
                "--search_mode vector를 사용하거나 --parse --ingest로 컬렉션을 다시 생성하세요."
            ) from e
        raise

    return _format_results(results)

def hybrid_search(collection_name, embed_provider, query, top_k, score_threshold=None):
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
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=top_k * 2,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        )
    except UnexpectedResponse as e:
        if not _is_missing_vector_name_error(e, "dense", "sparse"):
            raise
        results = _query_dense_points_with_fallback(
            collection_name,
            dense_vector,
            top_k,
            score_threshold,
        )

    return _format_results(results)
