from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text

def retrieve(collection_name: str, embed_provider: str, query: str, top_k=3):
    query_vector = embed_text(query, provider=embed_provider)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )

    formatted = []

    for r in results.points:

        formatted.append({
            "score": r.score,        
            **r.payload             
        })

    filtered = [d for d in formatted if d["score"] > 0.2]

    return filtered