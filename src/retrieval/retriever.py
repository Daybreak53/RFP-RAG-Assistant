from qdrant_client import models
from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text
from src.embeddings.sparse_embed import embed_sparse_text 

def retrieve(collection_name, embed_provider, query, top_k=3, score_threshold=0.7, search_mode="hybrid"):
    if search_mode == "vector":
        return vector_search(collection_name, embed_provider, query, top_k, score_threshold)
    elif search_mode == "keyword":
        return keyword_search(collection_name, query, top_k)
    elif search_mode == "hybrid":
        return hybrid_search(collection_name, embed_provider, query, top_k)
    else:
        raise ValueError(f"지원하지 않는 search_mode: '{search_mode}'")

def vector_search(collection_name, embed_provider, query, top_k, score_threshold):
    dense_vector = embed_text(query, provider=embed_provider)

    results = client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using="dense",
        limit=top_k,
        score_threshold=score_threshold
    )

    return [{"point_id": r.id, "score": r.score, **r.payload} for r in results.points]

def keyword_search(collection_name, query, top_k):
    sparse_vector = embed_sparse_text(query)

    results = client.query_points(
        collection_name=collection_name,
        query=sparse_vector,
        using="sparse",
        limit=top_k
    )

    return [{"point_id": r.id, "score": r.score, **r.payload} for r in results.points]

def hybrid_search(collection_name, embed_provider, query, top_k):
    dense_vector = embed_text(query, provider=embed_provider)
    sparse_vector = embed_sparse_text(query)

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

    return [{"point_id": r.id, "score": r.score, **r.payload} for r in results.points]