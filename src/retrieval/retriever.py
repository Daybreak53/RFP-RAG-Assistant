from typing import Optional
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text
from src.embeddings.sparse_embed import embed_sparse_text
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# HyDE / Multi-query용 텍스트 생성부
from src.generation.gen import generate_pure_text
from src.retrieval.reranker import rerank


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
    use_contextual: bool = False,
    use_multi_query: bool = False,
    multi_query_count: int = 5,
    multi_query_rrf_k: int = 60,
    candidate_k: Optional[int] = None,
    rerank_config: Optional[dict] = None,
):
    """
    search_mode:
      - vector
      - keyword
      - hybrid
      - mmr
      - hyde

    use_multi_query:
      - False: 기존 단일 query 검색
      - True: LLM이 query를 여러 개로 분해한 뒤 search_mode 방식으로 반복 검색
    """

    rerank_enabled = bool(rerank_config and rerank_config.get("enabled"))
    search_top_k = max(top_k, candidate_k or top_k) if rerank_enabled else top_k

    if use_multi_query:
        docs = multi_query_search(
            collection_name=collection_name,
            embed_provider=embed_provider,
            query=query,
            top_k=search_top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            base_search_mode=search_mode,
            query_count=multi_query_count,
            rrf_k=multi_query_rrf_k,
        )

    elif search_mode == "vector":
        docs = vector_search(
            collection_name,
            embed_provider,
            query,
            search_top_k,
            score_threshold,
            query_filter,
        )

    elif search_mode == "keyword":
        docs = keyword_search(
            collection_name,
            query,
            search_top_k,
            query_filter,
        )

    elif search_mode == "hybrid":
        docs = hybrid_search(
            collection_name,
            embed_provider,
            query,
            search_top_k,
            score_threshold,
            query_filter,
        )

    elif search_mode == "mmr":
        docs = mmr_search(
            collection_name,
            embed_provider,
            query,
            search_top_k,
            score_threshold,
            query_filter,
            lambda_param=0.95,
        )

    elif search_mode == "hyde":
        docs = hyde_search(
            collection_name,
            embed_provider,
            query,
            search_top_k,
            score_threshold,
            query_filter,
        )

    else:
        raise ValueError(f"지원하지 않는 search_mode: '{search_mode}'")

    if not rerank_enabled:
        return docs[:top_k]

    try:
        reranked = rerank(
            query=query,
            docs=docs,
            top_k=top_k,
            model_name=rerank_config.get("model"),
            max_length=rerank_config.get("max_length", 512),
            batch_size=rerank_config.get("batch_size", 16),
            max_content_chars=rerank_config.get("max_content_chars", 1800),
            score_threshold=rerank_config.get("score_threshold"),
            diversity_per_group=rerank_config.get("diversity_per_group", 1),
        )
    except Exception as exc:
        print(f"[경고] rerank 실패로 원본 검색 결과를 사용합니다: {exc} (config: {rerank_config})")
        return docs[:top_k]

    if not reranked:
        print(f"[경고] rerank 결과가 비어 원본 검색 결과를 사용합니다. (config: {rerank_config})")
        return docs[:top_k]

    return reranked


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
        query_filter,
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
            query_filter=query_filter,
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
                ),
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


def mmr_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: float = None,
    query_filter: Optional[models.Filter] = None,
    lambda_param: float = 0.5,
):
    dense_vector = embed_text(query, provider=embed_provider)

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k * 4,
            score_threshold=None,
            query_filter=query_filter,
            with_vectors=True,
            with_payload=True,
        )

    except UnexpectedResponse as e:
        if not _is_missing_vector_name_error(e, "dense"):
            raise

        results = client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            limit=top_k * 4,
            score_threshold=None,
            query_filter=query_filter,
            with_vectors=True,
            with_payload=True,
        )

    if not results or not hasattr(results, "points") or not results.points:
        return []

    points = results.points

    try:
        def _get_vector(p):
            if isinstance(p.vector, dict):
                return p.vector.get("dense") or list(p.vector.values())[0]
            return p.vector

        candidate_embeddings = np.array([_get_vector(p) for p in points])
        query_sims = np.array([p.score for p in points])

        if len(query_sims) > 1 and (query_sims.max() - query_sims.min()) > 1e-5:
            query_sims = (query_sims - query_sims.min()) / (
                query_sims.max() - query_sims.min()
            )

        selected_indices = []
        unselected_indices = list(range(len(points)))

        first_pick = int(np.argmax(query_sims))
        selected_indices.append(first_pick)
        unselected_indices.remove(first_pick)

        target_k = min(top_k, len(points))

        while len(selected_indices) < target_k and unselected_indices:
            mmr_scores = []

            for unsel_idx in unselected_indices:
                sim_to_query = query_sims[unsel_idx]

                sim_to_selected = max(
                    [
                        cosine_similarity(
                            [candidate_embeddings[unsel_idx]],
                            [candidate_embeddings[sel_idx]],
                        )[0][0]
                        for sel_idx in selected_indices
                    ]
                )

                sim_to_selected_scaled = (sim_to_selected + 1) / 2

                mmr_score = (
                    lambda_param * sim_to_query
                    - (1 - lambda_param) * sim_to_selected_scaled
                )

                mmr_scores.append((mmr_score, unsel_idx))

            best_idx = max(mmr_scores, key=lambda x: x[0])[1]
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        final_points = [points[i] for i in selected_indices]

    except Exception as e:
        print(f"[경고] MMR 연산 실패로 기본 유사도 순서로 대체합니다: {e}")
        final_points = points[:top_k]

    class MmrMappedResults:
        points = final_points

    return _format_results(MmrMappedResults())


def _generate_hypothetical_document(query: str, provider: str = "local") -> str:
    prompt = f"""당신은 정부 부처, 지자체 및 공공기관의 국가계약법 기반 제안요청서(RFP)와 조달청 입찰 공고문을 작성하는 20년 경력의 행정 전문가입니다.

[역할]
다음 유저의 질문(Query)을 읽고, 이 질문에 대한 정답 정보가 '실제 RFP 문서 본문(특히 평가 기준, 제안요청 사항, 계약 조건 등)'에 어떤 행정적/법적 문구 형태로 기술되어 있을지 예측하여 가상의 RFP 조항 1~2문장을 작성하세요.

[엄격한 제약 조건 - 반드시 준수할 것]
1. 절대 질문에 직접 대답(예: "~해야 합니다", "~입니다")하지 마십시오. 오직 RFP 본문 서술형 문장만 출력해야 합니다.
2. 구체적인 숫자(예: 3년, 100%, 4점, 5000만원 등)는 절대 임의로 지어내지 마십시오. 숫자가 들어갈 자리는 반드시 'OO%', 'X점', 'O개년', 'O원' 처럼 기호나 공란 기표로 대체하십시오.
3. 개조식 표기법 대신, 명사형 종결 어미(~할 것, ~에 의함, ~를 원칙으로 함)나 전문 행정 어조(~의거하여, ~를 충족하여야 함)를 사용하십시오.

유저 질문: {query}
가상 RFP 본문 조항:"""

    try:
        hypothetical_doc = generate_pure_text(prompt, provider=provider)
        return hypothetical_doc

    except Exception as e:
        print(f"[경고] HyDE 가상 문서 생성 실패로 원본 쿼리를 사용합니다. 에러: {e}")
        return query


def hyde_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: float = None,
    query_filter: Optional[models.Filter] = None,
):
    llm_provider = "openai" if embed_provider == "openai" else "local"

    hypothetical_doc = _generate_hypothetical_document(
        query,
        provider=llm_provider,
    )

    print(f"\n[HyDE 생성 문장]: {hypothetical_doc.strip()}\n")

    dense_vector = embed_text(hypothetical_doc, provider=embed_provider)

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k,
            score_threshold=None,
            query_filter=query_filter,
            with_payload=True,
        )

    except UnexpectedResponse as e:
        if not _is_missing_vector_name_error(e, "dense"):
            raise

        results = client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

    if not results or not hasattr(results, "points") or not results.points:
        class EmptyResults:
            points = []

        return _format_results(EmptyResults())

    return _format_results(results)


def _generate_multi_queries_by_llm(
    query: str,
    provider: str = "local",
    n: int = 5,
) -> list[str]:
    prompt = f"""
당신은 RFP/공공입찰 문서 검색 쿼리 생성 전문가입니다.

사용자 질문을 분석해서 벡터DB 검색에 적합한 검색 쿼리 {n}개를 생성하세요.

[중요]
- 질문이 여러 조건을 묻고 있다면 조건별로 분해하세요.
- 답변을 만들지 말고 검색어만 생성하세요.
- 원문 질문을 반드시 첫 번째 쿼리로 유지하세요.
- RFP, 제안요청서, 과업지시서, 입찰공고문에 실제로 등장할 법한 표현으로 바꾸세요.
- 번호, 설명, 따옴표 없이 한 줄에 하나씩 출력하세요.

[예시]
질문: 제안서 본문 내용의 최대 페이지 수 제한과 제안서 요약서의 최대 페이지 수 제한은 각각 얼마인가?

출력:
제안서 본문 내용의 최대 페이지 수 제한과 제안서 요약서의 최대 페이지 수 제한은 각각 얼마인가?
제안서 본문 최대 페이지 수 제한
제안서 본문 분량 제한
제안서 요약서 최대 페이지 수 제한
제안서 요약서 분량 제한
제안서 작성 기준 페이지 수

사용자 질문:
{query}
"""

    try:
        response = generate_pure_text(prompt, provider=provider)

        generated_queries = [
            line.strip("-•1234567890. ").strip()
            for line in response.splitlines()
            if line.strip()
        ]

        queries = [query] + generated_queries
        queries = list(dict.fromkeys(queries))

        return queries[:n]

    except Exception as e:
        print(f"[경고] LLM multi-query 생성 실패로 원본 쿼리만 사용합니다. 에러: {e}")
        return [query]


def _run_base_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: float,
    query_filter: Optional[models.Filter],
    base_search_mode: str,
):
    if base_search_mode == "vector":
        return vector_search(
            collection_name,
            embed_provider,
            query,
            top_k,
            score_threshold,
            query_filter,
        )

    elif base_search_mode == "keyword":
        return keyword_search(
            collection_name,
            query,
            top_k,
            query_filter,
        )

    elif base_search_mode == "hybrid":
        return hybrid_search(
            collection_name,
            embed_provider,
            query,
            top_k,
            score_threshold,
            query_filter,
        )

    elif base_search_mode == "mmr":
        return mmr_search(
            collection_name,
            embed_provider,
            query,
            top_k,
            score_threshold,
            query_filter,
            lambda_param=0.95,
        )

    elif base_search_mode == "hyde":
        return hyde_search(
            collection_name,
            embed_provider,
            query,
            top_k,
            score_threshold,
            query_filter,
        )

    else:
        raise ValueError(f"지원하지 않는 base_search_mode: '{base_search_mode}'")


def multi_query_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: float = None,
    query_filter: Optional[models.Filter] = None,
    base_search_mode: str = "hybrid",
    query_count: int = 5,
    rrf_k: int = 60,
):
    llm_provider = "openai" if embed_provider == "openai" else "local"

    queries = _generate_multi_queries_by_llm(
        query=query,
        provider=llm_provider,
        n=query_count,
    )

    print("\n[Multi-query 생성 쿼리]")
    for q in queries:
        print(f"- {q}")
    print()

    rrf_scores = {}

    for q in queries:
        docs = _run_base_search(
            collection_name=collection_name,
            embed_provider=embed_provider,
            query=q,
            top_k=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            base_search_mode=base_search_mode,
        )

        for rank, doc in enumerate(docs, start=1):
            key = (
                doc.get("doc_id"),
                doc.get("chunk_id"),
                doc.get("point_id"),
            )

            if key not in rrf_scores:
                rrf_scores[key] = {
                    "score": 0.0,
                    "doc": doc,
                    "best_rank": rank,
                    "match_count": 0,
                }

            rrf_scores[key]["score"] += 1.0 / (rrf_k + rank)
            rrf_scores[key]["best_rank"] = min(rrf_scores[key]["best_rank"], rank)
            rrf_scores[key]["match_count"] += 1

    final_docs = []
    for entry in sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True):
        updated_doc = dict(entry["doc"])
        updated_doc["rrf_score"] = entry["score"]
        updated_doc["score"] = entry["score"]
        updated_doc["best_rank"] = entry["best_rank"]
        updated_doc["matched_query_count"] = entry["match_count"]
        final_docs.append(updated_doc)

    return final_docs[:top_k]