import yaml
from dotenv import load_dotenv
from src.retrieval.retriever import retrieve

load_dotenv()

def compare_search_modes(query: str, config_path="config.yaml"):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    embed_provider = config["providers"]["embedding"]
    collection_name = config["collection_name"][embed_provider]
    top_k = config["retrieval"]["top_k"]

    modes = ["vector", "keyword", "hybrid"]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  검색 모드: [{mode.upper()}]  |  쿼리: {query}")
        print(f"{'='*60}")

        results = retrieve(
            collection_name=collection_name,
            embed_provider=embed_provider,
            query=query,
            top_k=top_k,
            search_mode=mode,
        )

        if not results:
            print("  검색 결과 없음")
            continue

        for rank, doc in enumerate(results, 1):
            print(f"\n  [{rank}위] score={doc.get('score', 0):.4f}")
            print(f"       파일  : {doc.get('file_name', 'N/A')}")
            print(f"       제목  : {doc.get('title', 'N/A')}")
            print(f"       섹션  : {doc.get('section_title', 'N/A')}")
            print(f"       내용  : {doc.get('content', '')[:80].strip()}...")

if __name__ == "__main__":
    import sys
    with open("config.yaml", encoding="utf-8") as f:
        import yaml
        _config = yaml.safe_load(f)
    query = sys.argv[1] if len(sys.argv) > 1 else _config.get("query", "")
    compare_search_modes(query)