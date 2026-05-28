from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from omegaconf import DictConfig, OmegaConf
from langfuse import get_client

from src.parsing.meta_db import normalize_filename
from src.retrieval.retriever import retrieve_with_candidates
from src.retrieval.filter_extractor import MetadataFilter, resolve_filter
from src.retrieval.query_router import route

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ──────────────────────────────────────────────────────────────
# 데이터 모델
# ──────────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    """
    평가 데이터셋의 한 레코드를 내부 표준 형태로 정규화한 것.
    relevance_type 은 현재 "file" 고정, 추후 "chunk" 확장 여지.
    """
    user_input:      str
    reference:       str
    relevant_ids:    List[str]                                 # 정규화된 파일 키 (normalize_filename 적용)
    relevance_type:  Literal["file", "chunk"] = "file"
    raw_file_name:   str = ""                                  # 데이터셋 원본 파일명 (디버깅용)
    comment:         str = ""


# ──────────────────────────────────────────────────────────────
# 데이터셋 로더
# ──────────────────────────────────────────────────────────────

def load_eval_dataset(dataset_path: Path) -> List[EvalRecord]:
    """
    sample.json 형식 로더.
    각 record: {"file_name", "user_input", "reference", "comment"}
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"평가 데이터셋이 없습니다: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records: List[EvalRecord] = []
    for idx, item in enumerate(raw):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            logger.warning(f"[{idx}] file_name 누락 — 레코드 건너뜀")
            continue

        records.append(EvalRecord(
            user_input     = str(item.get("user_input", "")).strip(),
            reference      = str(item.get("reference", "")).strip(),
            relevant_ids   = [normalize_filename(file_name)],
            relevance_type = "file",
            raw_file_name  = file_name,
            comment        = str(item.get("comment", "")).strip(),
        ))

    logger.info(f"평가 데이터셋 로드 완료: {len(records)}건 ({dataset_path})")
    return records


# ──────────────────────────────────────────────────────────────
# 지표 함수 (binary relevance)
# ──────────────────────────────────────────────────────────────

def precision_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
    if k <= 0 or not predicted:
        return 0.0
    hits = sum(1 for p in predicted[:k] if p in relevant)
    return hits / k


def recall_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for p in predicted[:k] if p in relevant)
    return hits / len(relevant)


def mrr_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
    if k <= 0 or not predicted or not relevant:
        return 0.0
    for i, p in enumerate(predicted[:k], start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
    if k <= 0 or not predicted or not relevant:
        return 0.0

    gains = [1.0 if p in relevant else 0.0 for p in predicted[:k]]
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# 단일 record 평가
# ──────────────────────────────────────────────────────────────

def _doc_to_id(doc: Dict[str, Any]) -> str:
    """Qdrant 결과 doc 에서 평가 비교용 정규화 file 키 추출."""
    return normalize_filename(str(doc.get("file_name", "")))


def _predicted_ids(docs: List[Dict[str, Any]]) -> List[str]:
    """
    file-level 평가용: 검색 결과의 file_name을 정규화 + 순서 보존 dedupe 하여
    unique file 리스트 반환. 같은 파일의 여러 chunk는 첫 등장 rank만 유지.
    빈 file_name 은 평가 대상에서 제외.
    """
    seen: set[str] = set()
    result: List[str] = []
    for d in docs:
        fid = _doc_to_id(d)
        if not fid or fid in seen:
            continue
        seen.add(fid)
        result.append(fid)
    return result


def _rerank_status(final_docs: List[Dict[str, Any]], rerank_enabled: bool) -> str:
    """
    평가용 휴리스틱: rerank 가 실제 적용됐는지 판정.
    reranker.py 가 rerank_score 키를 doc 에 부여하는 점을 이용.

    반환값:
      - "disabled":      rerank 비활성 설정
      - "applied":       rerank 정상 적용됨
      - "fallback":      rerank 활성이지만 실패/빈 결과 → base 검색으로 fallback 발생
      - "empty_result":  final_docs 자체가 비어있음
    """
    if not rerank_enabled:
        return "disabled"
    if not final_docs:
        return "empty_result"
    if any("rerank_score" in d for d in final_docs):
        return "applied"
    return "fallback"


def evaluate_record(
    record:         EvalRecord,
    final_docs:     List[Dict[str, Any]],
    candidates:     List[Dict[str, Any]],
    top_k:          int,
    candidate_k:    int,
    route_type:     str,
    rerank_enabled: bool,
) -> Dict[str, Any]:
    """
    한 record 에 대한 모든 지표 계산.
    """
    final_ids     = _predicted_ids(final_docs)
    candidate_ids = _predicted_ids(candidates)
    relevant      = record.relevant_ids
    rerank_state  = _rerank_status(final_docs, rerank_enabled)

    return {
        "route_type":           route_type,
        "user_input":           record.user_input,
        "relevant_ids":         relevant,
        "final_ids":            final_ids[:top_k],
        "candidate_ids":        candidate_ids[:candidate_k],
        "rerank_status":        rerank_state,
        "metrics": {
            f"precision@{top_k}":        precision_at_k(final_ids, relevant, top_k),
            f"recall@{top_k}":           recall_at_k(final_ids, relevant, top_k),
            f"mrr@{top_k}":              mrr_at_k(final_ids, relevant, top_k),
            f"ndcg@{top_k}":             ndcg_at_k(final_ids, relevant, top_k),
            f"recall@{candidate_k}":     recall_at_k(candidate_ids, relevant, candidate_k),
        },
    }


_RERANK_STATUS_VALUES = ("applied", "fallback", "empty_result")


def aggregate_metrics(per_record: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    모든 record 의 metric 평균 + rerank status 비율.
    disabled 상태는 per-record 진단값으로만 남기고 aggregate 에서는 runtime.rerank_enabled 로 확인한다.
    """
    if not per_record:
        return {}

    total = len(per_record)
    metric_keys = list(per_record[0]["metrics"].keys())
    agg: Dict[str, float] = {}

    for key in metric_keys:
        vals = [r["metrics"][key] for r in per_record]
        agg[key] = sum(vals) / total

    statuses = [r.get("rerank_status", "disabled") for r in per_record]
    for status_value in _RERANK_STATUS_VALUES:
        agg[f"rerank_{status_value}_ratio"] = sum(1 for s in statuses if s == status_value) / total

    return agg


# ──────────────────────────────────────────────────────────────
# 메인 엔트리
# ──────────────────────────────────────────────────────────────

def _build_explicit_filter(cfg: DictConfig) -> MetadataFilter:
    f = cfg.filter
    return MetadataFilter(
        organization        = f.org,
        budget_min          = f.budget_min,
        budget_max          = f.budget_max,
        announcement_after  = f.announce_after,
        announcement_before = f.announce_before,
        bid_start_after     = f.bid_start_after,
        bid_deadline_before = f.bid_deadline_before,
        title_keyword       = f.title,
        doc_id              = f.doc_id,
    )


def _router_dict(cfg: DictConfig) -> Optional[dict]:
    if not hasattr(cfg, "router"):
        return None
    return OmegaConf.to_container(cfg.router, resolve=True)


def _push_langfuse_scores(
    langfuse_client: Any,
    aggregate: Dict[str, float],
    dataset_name: str,
) -> None:
    """
    aggregate score 를 Langfuse 에 fail-soft 로 등록.
    실패해도 평가 흐름은 깨지지 않게 try/except 로 보호.
    """
    try:
        with langfuse_client.start_as_current_observation(
            name     = "retrieval_eval_summary",
            as_type  = "span",
            input    = {"dataset": dataset_name},
            metadata = {"record_count_metric_keys": list(aggregate.keys())},
        ) as span:
            span.update(output=aggregate)
            trace_id = getattr(span, "trace_id", None)

        if not trace_id:
            print("[경고] Langfuse trace_id 누락으로 score 등록 건너뜀")
            return

        for name, value in aggregate.items():
            try:
                langfuse_client.create_score(
                    trace_id  = str(trace_id),
                    name      = f"retrieval/{name}",
                    value     = float(value),
                    data_type = "NUMERIC",
                    comment   = f"dataset={dataset_name}",
                )
            except Exception as score_exc:
                print(f"[경고] Langfuse score 등록 실패 ({name}): {score_exc}")

        langfuse_client.flush()
        logger.info("Langfuse aggregate score 등록 완료")

    except Exception as exc:
        print(f"[경고] Langfuse 통합 실패로 score 등록을 건너뜁니다: {exc}")


def _save_results_json(
    output_dir:    Path,
    aggregate:     Dict[str, float],
    per_record:    List[Dict[str, Any]],
    runtime_meta:  Dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / f"retrieval_eval_{timestamp}.json"

    payload = {
        "timestamp":  timestamp,
        "runtime":    runtime_meta,
        "aggregate":  aggregate,
        "per_record": per_record,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"평가 결과 저장: {out_path}")
    return out_path


def _print_summary(aggregate: Dict[str, float], record_count: int, out_path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"📊 Retrieval 평가 결과 (총 {record_count}건)")
    print("=" * 60)
    for key, val in aggregate.items():
        print(f"  {key:<28} {val:.4f}")
    print(f"\n  저장: {out_path}")
    print("=" * 60 + "\n")


def run_retrieval_eval(cfg: DictConfig) -> Dict[str, Any]:
    """
    검색 단계 정량 평가 엔트리.
    cfg.evaluation.retrieval_metrics 섹션의 dataset_path / output_dir / top_k / candidate_k / langfuse_enabled 사용.
    검색 파라미터(retrieval/router/filter) 는 일반 RAG 호출과 동일하게 cfg 기반으로 구성.
    """
    rm_cfg          = cfg.evaluation.retrieval_metrics
    dataset_path    = (PROJECT_ROOT / rm_cfg.dataset_path).resolve()
    output_dir      = (PROJECT_ROOT / rm_cfg.output_dir).resolve()
    top_k           = int(rm_cfg.top_k)
    candidate_k     = int(rm_cfg.candidate_k)
    use_langfuse    = bool(rm_cfg.get("langfuse_enabled", False))

    embed_provider  = cfg.providers.embedding
    llm_provider    = cfg.providers.llm
    llm_model_name  = cfg.providers.models.llm[llm_provider]
    collection_name = cfg.vector_db.collection_names[embed_provider]

    rerank_config   = OmegaConf.to_container(cfg.retrieval.rerank, resolve=True)
    rerank_enabled  = bool(rerank_config and rerank_config.get("enabled"))
    explicit_filter = _build_explicit_filter(cfg)
    auto_extract    = not cfg.filter.no_auto

    router_section  = _router_dict(cfg)
    use_router      = router_section.get("enabled", True) if router_section else True
    use_llm_cls     = router_section.get("use_llm_classifier", False) if router_section else False
    force_type      = router_section.get("force_query_type") if router_section else None

    records = load_eval_dataset(dataset_path)
    if not records:
        print("[경고] 평가 데이터셋이 비어있어 평가를 종료합니다.")
        return {}

    langfuse_client = get_client() if use_langfuse else None
    per_record: List[Dict[str, Any]] = []

    for idx, record in enumerate(records, start=1):
        logger.info(f"[{idx}/{len(records)}] 평가 중: {record.user_input[:60]}...")

        # 라우터 결정 — use_router=False 면 cfg 기본값 사용
        if use_router:
            route_cfg = route(
                query              = record.user_input,
                use_llm_classifier = use_llm_cls,
                llm_provider       = llm_provider,
                llm_model          = llm_model_name,
                router_cfg         = router_section,
                force_query_type   = force_type,
            )
            eff_search_mode      = route_cfg.search_mode
            eff_top_k            = route_cfg.top_k
            eff_threshold        = route_cfg.score_threshold
            eff_use_multi_query  = route_cfg.use_multi_query
            route_type           = route_cfg.query_type.value
        else:
            eff_search_mode      = cfg.retrieval.search_mode
            eff_top_k            = cfg.retrieval.top_k
            eff_threshold        = cfg.retrieval.score_threshold
            eff_use_multi_query  = cfg.retrieval.multi_query.enabled
            route_type           = "unknown"

        qdrant_filter = resolve_filter(
            query           = record.user_input,
            explicit_filter = explicit_filter,
            auto_extract    = auto_extract,
            query_type      = route_type,
        )

        # 평가 metric 의 top_k 는 cfg.evaluation.retrieval_metrics.top_k 가 우선.
        # router의 top_k 는 무시 (평가는 고정 컷오프로 비교해야 의미 있음).
        try:
            final_docs, candidates = retrieve_with_candidates(
                collection_name   = collection_name,
                embed_provider    = embed_provider,
                query             = record.user_input,
                top_k             = top_k,
                candidate_k       = candidate_k,
                score_threshold   = eff_threshold,
                search_mode       = eff_search_mode,
                query_filter      = qdrant_filter,
                use_multi_query   = eff_use_multi_query,
                multi_query_count = cfg.retrieval.multi_query.query_count,
                multi_query_rrf_k = cfg.retrieval.get("multi_query_rrf_k", 60),
                rerank_config     = rerank_config,
            )
        except Exception as exc:
            print(f"[경고] [{idx}] 검색 실패로 빈 결과 사용: {exc}")
            final_docs, candidates = [], []

        per_record.append(evaluate_record(
            record         = record,
            final_docs     = final_docs,
            candidates     = candidates,
            top_k          = top_k,
            candidate_k    = candidate_k,
            route_type     = route_type,
            rerank_enabled = rerank_enabled,
        ))

    aggregate = aggregate_metrics(per_record)

    runtime_meta = {
        "dataset_path":     str(dataset_path),
        "record_count":     len(records),
        "top_k":            top_k,
        "candidate_k":      candidate_k,
        "embed_provider":   embed_provider,
        "collection_name":  collection_name,
        "search_mode":      cfg.retrieval.search_mode,
        "rerank_enabled":   rerank_enabled,
        "router_enabled":   use_router,
        "use_llm_classifier": use_llm_cls,
    }

    out_path = _save_results_json(output_dir, aggregate, per_record, runtime_meta)
    _print_summary(aggregate, len(records), out_path)

    if use_langfuse and langfuse_client is not None:
        _push_langfuse_scores(langfuse_client, aggregate, dataset_path.name)

    return {"aggregate": aggregate, "per_record": per_record, "output_path": str(out_path)}
