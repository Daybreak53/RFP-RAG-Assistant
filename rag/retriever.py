from rag.vectordb import client
from rag.embedding import embed_text
from rag.config import EMBEDDING_PROVIDER, COLLECTION_MAP

def retrieve(query, top_k=3):

    collection = COLLECTION_MAP[EMBEDDING_PROVIDER]
    query_vector = embed_text(query, provider=EMBEDDING_PROVIDER)

    results = client.query_points(
        collection_name=collection,
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