import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text
from src.embeddings.sparse_embed import embed_sparse_text
from src.generation.gen import generate_pure_text

# 로거 설정
logger = logging.getLogger(__name__)


def _format_points(points: Iterable[Any]) -> List[Dict[str, Any]]:
    """
    Qdrant 검색 결과(Point) 목록을 내부 파이프라인에서 사용하는 Dict 리스트 포맷으로 변환
    """
    if not points:
        return []
    
    return [
        {
            "point_id": p.id,
            "score": getattr(p, "score", 0.0),
            **(getattr(p, "payload", {}) or {})
        }
        for p in points
    ]


def _is_missing_vector_name_error(error: Exception, *vector_names: str) -> bool:
    """
    Qdrant 예외가 특정 벡터 이름 누락으로 인한 것인지 확인
    """
    message = str(error)
    return (
        "Not existing vector name" in message
        and any(v_name in message for v_name in vector_names)
    )


def _query_dense_points(
    collection_name: str,
    dense_vector: List[float],
    top_k: int,
    score_threshold: float,
    query_filter: Optional[models.Filter] = None,
    using_dense: bool = True,
) -> Any:
    """
    단일 Dense 벡터 검색 수행 (using 파라미터 제어)
    """
    query_kwargs = {
        "collection_name": collection_name,
        "query": dense_vector,
        "limit": top_k,
        "score_threshold": score_threshold,
        "query_filter": query_filter,
    }
    if using_dense:
        query_kwargs["using"] = "dense"

    return client.query_points(**query_kwargs)


def _query_dense_points_with_fallback(
    collection_name: str,
    dense_vector: List[float],
    top_k: int,
    score_threshold: Optional[float] = None,
    query_filter: Optional[models.Filter] = None,
) -> Any:
    """
    이름이 지정된 dense 벡터가 없을 경우 기본 벡터로 Fallback 검색 수행
    """
    try:
        return _query_dense_points(
            collection_name, dense_vector, top_k, score_threshold, query_filter, using_dense=True
        )
    except UnexpectedResponse as e:
        if not _is_missing_vector_name_error(e, "dense"):
            raise
        # 'dense' 이름이 없으면 using 파라미터 없이 재시도
        return _query_dense_points(
            collection_name, dense_vector, top_k, score_threshold, query_filter, using_dense=False
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
) -> List[Dict[str, Any]]:
    """
    질의 의도 및 설정에 따라 적절한 검색 모드로 라우팅
    """
    logger.debug(f"검색 시작 | 모드: {search_mode} | 쿼리: {query}")

    if use_multi_query:
        return multi_query_search(
            collection_name=collection_name,
            embed_provider=embed_provider,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            base_search_mode=search_mode,
            query_count=multi_query_count,
        )

    if search_mode == "vector":
        return vector_search(collection_name, embed_provider, query, top_k, score_threshold, query_filter)
    elif search_mode == "keyword":
        results = keyword_search(collection_name, query, top_k, query_filter)
        # keyword 결과가 없으면 vector로 재검색
        if not results:
            logger.info("keyword 검색 결과 없음 → vector 검색으로 폴백")
            results = vector_search(
                collection_name, embed_provider, query, top_k, score_threshold, query_filter
            )
        return results
    elif search_mode == "hybrid":
        return hybrid_search(collection_name, embed_provider, query, top_k, score_threshold, query_filter)
    elif search_mode == "mmr":
        return mmr_search(collection_name, embed_provider, query, top_k, score_threshold, query_filter)
    elif search_mode == "hyde":
        return hyde_search(collection_name, embed_provider, query, top_k, score_threshold, query_filter)
    else:
        raise ValueError(f"지원하지 않는 search_mode: '{search_mode}'")


def vector_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: Optional[float] = None,
    query_filter: Optional[models.Filter] = None,
) -> List[Dict[str, Any]]:
    """
    단일 Dense Vector 기반 검색
    """
    dense_vector = embed_text(query, provider=embed_provider)
    results = _query_dense_points_with_fallback(
        collection_name, dense_vector, top_k, score_threshold, query_filter
    )
    return _format_points(results.points)


def keyword_search(
    collection_name: str,
    query: str,
    top_k: int,
    query_filter: Optional[models.Filter] = None,
) -> List[Dict[str, Any]]:
    """
    BM25/Sparse Vector 기반 키워드 검색
    """
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
                "Vector 모드를 사용하거나 DB를 다시 적재하세요."
            ) from e
        raise

    return _format_points(results.points)


def hybrid_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: Optional[float] = None,
    query_filter: Optional[models.Filter] = None,
) -> List[Dict[str, Any]]:
    """
    Dense + Sparse 하이브리드 검색 (RRF Fusion)
    """
    dense_vector = embed_text(query, provider=embed_provider)
    sparse_vector = embed_sparse_text(query)

    try:
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector, using="dense", limit=top_k * 2, filter=query_filter
                ),
                models.Prefetch(
                    query=sparse_vector, using="sparse", limit=top_k * 2, filter=query_filter
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            query_filter=query_filter,
        )
    except UnexpectedResponse as e:
        if not _is_missing_vector_name_error(e, "dense", "sparse"):
            raise
        logger.warning("하이브리드 검색 실패: Sparse/Dense 벡터 누락. 일반 벡터 검색으로 Fallback 합니다.")
        results = _query_dense_points_with_fallback(
            collection_name, dense_vector, top_k, score_threshold, query_filter
        )

    return _format_points(results.points)


def mmr_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: Optional[float] = None,
    query_filter: Optional[models.Filter] = None,
    lambda_param: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    다양성을 고려한 MMR(Maximal Marginal Relevance) 검색
    """
    dense_vector = embed_text(query, provider=embed_provider)

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k * 4,  # MMR 계산을 위해 더 많은 후보군 검색
            score_threshold=score_threshold,
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
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_vectors=True,
            with_payload=True,
        )

    if not results or not getattr(results, "points", None):
        return []

    points = results.points

    try:
        def _get_vector(p: Any) -> List[float]:
            if isinstance(p.vector, dict):
                return p.vector.get("dense") or list(p.vector.values())[0]
            return p.vector

        candidate_embeddings = np.array([_get_vector(p) for p in points])
        query_sims = np.array([p.score for p in points])

        # Min-Max Scaling
        if len(query_sims) > 1 and (query_sims.max() - query_sims.min()) > 1e-5:
            query_sims = (query_sims - query_sims.min()) / (query_sims.max() - query_sims.min())

        sim_matrix = cosine_similarity(candidate_embeddings)

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
                sim_to_selected = max(sim_matrix[unsel_idx, sel_idx] for sel_idx in selected_indices)
                sim_to_selected_scaled = (sim_to_selected + 1) / 2
                mmr_score = (lambda_param * sim_to_query) - ((1 - lambda_param) * sim_to_selected_scaled)
                mmr_scores.append((mmr_score, unsel_idx))

            best_idx = max(mmr_scores, key=lambda x: x[0])[1]
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        final_points = [points[i] for i in selected_indices]

    except Exception as e:
        logger.warning(f"MMR 연산 실패로 기본 유사도 순서로 대체합니다: {e}")
        final_points = points[:top_k]

    return _format_points(final_points)


def hyde_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: Optional[float] = None,
    query_filter: Optional[models.Filter] = None,
) -> List[Dict[str, Any]]:
    """
    가상 문서를 생성하여 검색하는 HyDE 알고리즘 적용
    """
    llm_provider = "openai" if embed_provider == "openai" else "local"

    prompt = f"""당신은 정부 부처, 지자체 및 공공기관의 국가계약법 기반 제안요청서(RFP)와 조달청 입찰 공고문을 작성하는 20년 경력의 행정 전문가입니다.

[역할]
다음 유저의 질문(Query)을 읽고, 이 질문에 대한 정답 정보가 '실제 RFP 문서 본문'에 어떤 행정적/법적 문구 형태로 기술되어 있을지 예측하여 가상의 RFP 조항 1~2문장을 작성하세요.

[엄격한 제약 조건]
1. 질문에 직접 대답하지 마십시오. 오직 RFP 본문 서술형 문장만 출력해야 합니다.
2. 구체적인 숫자(예: 3년, 100%)는 절대 임의로 지어내지 마십시오. 숫자가 들어갈 자리는 'OO%', 'X점', 'O원' 처럼 대체하십시오.
3. 명사형 종결 어미나 전문 행정 어조를 사용하십시오.

유저 질문: {query}
가상 RFP 본문 조항:"""

    try:
        hypothetical_doc = generate_pure_text(prompt, provider=llm_provider)
        logger.info(f"HyDE 생성 문장: {hypothetical_doc.strip()}")
    except Exception as e:
        logger.warning(f"HyDE 가상 문서 생성 실패로 원본 쿼리를 사용합니다: {e}")
        hypothetical_doc = query

    dense_vector = embed_text(hypothetical_doc, provider=embed_provider)
    
    try:
        results = client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k,
            score_threshold=score_threshold,
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

    return _format_points(getattr(results, "points", []))


def multi_query_search(
    collection_name: str,
    embed_provider: str,
    query: str,
    top_k: int,
    score_threshold: Optional[float] = None,
    query_filter: Optional[models.Filter] = None,
    base_search_mode: str = "hybrid",
    query_count: int = 5,
) -> List[Dict[str, Any]]:
    """
    질의를 LLM을 통해 여러 개로 파생시켜 검색 후 병합(RRF/중복제거)하는 Multi-Query Retrieval
    """
    llm_provider = "openai" if embed_provider == "openai" else "local"

    prompt = f"""
당신은 RFP/공공입찰 문서 검색 쿼리 생성 전문가입니다.
사용자 질문을 분석해서 벡터DB 검색에 적합한 검색 쿼리 {query_count}개를 생성하세요.

[중요]
- 질문이 여러 조건을 묻고 있다면 조건별로 분해하세요.
- 답변을 만들지 말고 검색어만 생성하세요.
- 원문 질문을 반드시 첫 번째 쿼리로 유지하세요.
- RFP, 제안요청서 등에 실제로 등장할 법한 표현으로 바꾸세요.
- 번호, 설명, 따옴표 없이 한 줄에 하나씩 출력하세요.

사용자 질문:
{query}
"""

    try:
        response = generate_pure_text(prompt, provider=llm_provider)
        generated_queries = [
            line.strip("-•1234567890. ").strip()
            for line in response.splitlines()
            if line.strip()
        ]
        queries = list(dict.fromkeys([query] + generated_queries))[:query_count]
    except Exception as e:
        logger.warning(f"LLM multi-query 생성 실패로 원본 쿼리만 사용합니다: {e}")
        queries = [query]

    logger.info(f"Multi-query 생성 쿼리 목록: {queries}")

    all_docs = []
    
    # 생성된 각각의 쿼리로 base_search 수행
    for q in queries:
        docs = retrieve(
            collection_name=collection_name,
            embed_provider=embed_provider,
            query=q,
            top_k=top_k,
            score_threshold=score_threshold,
            search_mode=base_search_mode,
            query_filter=query_filter,
            use_multi_query=False  # 무한 루프 방지
        )
        all_docs.extend(docs)

    # 중복 제거 (가장 높은 점수를 기준으로 병합)
    unique_docs = {}
    for doc in all_docs:
        key = (doc.get("doc_id"), doc.get("chunk_id"), doc.get("point_id"))
        if key not in unique_docs or doc.get("score", 0) > unique_docs[key].get("score", 0):
            unique_docs[key] = doc

    # 점수순으로 내림차순 정렬하여 top_k 개수만큼 반환
    final_docs = sorted(unique_docs.values(), key=lambda x: x.get("score", 0), reverse=True)
    return final_docs[:top_k]