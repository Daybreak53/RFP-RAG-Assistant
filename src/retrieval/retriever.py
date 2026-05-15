from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text

def retrieve(collection_name, embed_provider, query, top_k=3, score_threshold=0.2, search_mode="vector"):
    if search_mode == "vector":
        return vector_search(collection_name, embed_provider, query, top_k, score_threshold)
    elif search_mode == "hybrid":
        return hybrid_search(collection_name, embed_provider, query, top_k, score_threshold)
    else:
        raise ValueError(f"지원하지 않는 search_mode: '{search_mode}'")

def vector_search(collection_name, embed_provider, query, top_k, score_threshold):
    query_vector = embed_text(query, provider=embed_provider)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )

    formatted = [{"score": r.score, **r.payload} for r in results.points]
    filtered = [d for d in formatted if d["score"] > score_threshold]

    return filtered

def hybrid_search(collection_name, embed_provider, query, top_k, score_threshold):
    # TODO: 키워드 검색 결과 + 벡터 검색 결과 결합 (RRF 등)
    raise NotImplementedError("하이브리드 검색 구현 필요")