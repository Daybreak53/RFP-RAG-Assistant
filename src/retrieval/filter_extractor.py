from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from qdrant_client import models

@dataclass
class MetadataFilter:
    """검색에 사용할 메타데이터 필터 조건 집합"""
    organization:        Optional[str]   = None   # 발주 기관 (부분 일치)
    budget_min:          Optional[float] = None   # 최소 예산 (만원)
    budget_max:          Optional[float] = None   # 최대 예산 (만원)
    announcement_after:  Optional[str]   = None   # 공고일 시작 (YYYY-MM-DD)
    announcement_before: Optional[str]   = None   # 공고일 종료 (YYYY-MM-DD)
    bid_start_after:     Optional[str]   = None   # 입찰 시작일 이후
    bid_deadline_before: Optional[str]   = None   # 입찰 마감일 이전
    title_keyword:       Optional[str]   = None   # 사업명 키워드 (부분 일치)
    doc_id:              Optional[str]   = None   # 공고 번호 (정확 일치)

    def is_empty(self) -> bool:
        return all(v is None for v in self.__dict__.values())

# 예산 단위 변환 테이블 (모두 만원 기준으로 통일)
_BUDGET_UNIT: dict[str, float] = {
    "억": 10_000,
    "천만": 1_000,
    "백만": 100,
    "만": 1,
}

# 기관 유형 키워드
_ORG_SUFFIXES = (
    "시", "군", "구", "도", "청", "처", "원", "부", "청",
    "공사", "공단", "재단", "협회", "센터", "연구원", "연구소",
    "교육청", "경찰청", "소방서", "보건소",
)

def _extract_budget(text: str) -> tuple[Optional[float], Optional[float]]:
    """
    예: "50억 이상"  →  (500000, None)
        "10억 ~ 20억"  →  (100000, 200000)
        "30억 이하"  →  (None, 300000)
    """
    budget_min = budget_max = None

    # "X억 ~ Y억" 또는 "X억에서 Y억" 패턴
    range_pat = re.compile(
        r"(\d+(?:\.\d+)?)\s*([억천백만]+)\s*(?:~|에서|부터|∼)\s*(\d+(?:\.\d+)?)\s*([억천백만]+)"
    )
    m = range_pat.search(text)
    if m:
        val1 = float(m.group(1)) * _unit_to_manwon(m.group(2))
        val2 = float(m.group(3)) * _unit_to_manwon(m.group(4))
        return min(val1, val2), max(val1, val2)

    # "X억 이상/초과" 패턴
    over_pat = re.compile(r"(\d+(?:\.\d+)?)\s*([억천백만]+)\s*(이상|초과|넘는|이상의)")
    m = over_pat.search(text)
    if m:
        budget_min = float(m.group(1)) * _unit_to_manwon(m.group(2))

    # "X억 이하/미만" 패턴
    under_pat = re.compile(r"(\d+(?:\.\d+)?)\s*([억천백만]+)\s*(이하|미만|이내|이하의)")
    m = under_pat.search(text)
    if m:
        budget_max = float(m.group(1)) * _unit_to_manwon(m.group(2))

    return budget_min, budget_max


def _unit_to_manwon(unit_str: str) -> float:
    """단위 문자열 → 만원 환산 배율"""
    for key, val in _BUDGET_UNIT.items():
        if key in unit_str:
            return val
    return 1.0


def _extract_organization(text: str) -> Optional[str]:
    """
    예: "서울특별시 발주", "행정안전부에서", "한국도로공사의"
    """
    # "X에서 발주/공고", "X의 RFP" 등 기관 언급 패턴
    patterns = [
        r"([\w가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)?\s*[\w가-힣]*(?:시|군|구|청|처|부|원|공사|공단|재단|연구원|연구소|협회|센터|교육청)?)\s*(?:에서|의|이|가|에서\s*발주|에서\s*공고|발주)",
        r"([\w가-힣]{2,15}(?:청|처|부|원|공사|공단|재단|연구원|연구소|협회|센터|교육청|경찰청|소방서|보건소))",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1).strip()
            # 너무 짧거나 일반 단어면 제외
            if len(candidate) >= 3:
                return candidate
    return None


def _extract_date(text: str, keywords: list[str]) -> Optional[str]:
    """
    특정 키워드 근처의 날짜(YYYY-MM-DD 또는 YYYY.MM.DD) 추출
    """
    date_pat = re.compile(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})")

    for kw in keywords:
        idx = text.find(kw)
        if idx == -1:
            continue
        snippet = text[max(0, idx - 10): idx + 50]
        m = date_pat.search(snippet)
        if m:
            y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
            return f"{y}-{mo}-{d}"
    return None


def extract_filters_from_query(query: str) -> MetadataFilter:
    """
    자연어 쿼리에서 메타데이터 필터를 자동 추출
    """
    flt = MetadataFilter()

    flt.organization = _extract_organization(query)

    flt.budget_min, flt.budget_max = _extract_budget(query)

    flt.announcement_after = _extract_date(
        query, ["공고일", "공개일", "공고 이후", "공고 후"]
    )
    flt.bid_deadline_before = _extract_date(
        query, ["마감", "마감일", "입찰 마감", "제출 기한"]
    )
    flt.bid_start_after = _extract_date(
        query, ["입찰 시작", "시작일", "접수 시작"]
    )

    return flt

def build_qdrant_filter(flt: MetadataFilter) -> Optional[models.Filter]:
    """
    매핑:
        organization        → MatchText (부분 일치)
        budget_min/max      → Range
        announcement_after/before → Range (문자열 날짜 사전순 비교)
        bid_start_after     → Range
        bid_deadline_before → Range
        title_keyword       → MatchText
        doc_id              → MatchValue (정확 일치)
    """
    if flt is None or flt.is_empty():
        return None

    must: list[models.Condition] = []

    # --- 발주 기관 (부분 일치) ---
    if flt.organization:
        must.append(
            models.FieldCondition(
                key="organization",
                match=models.MatchText(text=flt.organization),
            )
        )

    # --- 예산 범위 ---
    if flt.budget_min is not None or flt.budget_max is not None:
        must.append(
            models.FieldCondition(
                key="budget",
                range=models.Range(
                    gte=flt.budget_min,
                    lte=flt.budget_max,
                ),
            )
        )

    # --- 공고일 범위 ---
    if flt.announcement_after or flt.announcement_before:
        must.append(
            models.FieldCondition(
                key="announcement_date",
                range=models.Range(
                    gte=flt.announcement_after,
                    lte=flt.announcement_before,
                ),
            )
        )

    # --- 입찰 시작일 이후 ---
    if flt.bid_start_after:
        must.append(
            models.FieldCondition(
                key="bid_start",
                range=models.Range(gte=flt.bid_start_after),
            )
        )

    # --- 입찰 마감일 이전 ---
    if flt.bid_deadline_before:
        must.append(
            models.FieldCondition(
                key="bid_deadline",
                range=models.Range(lte=flt.bid_deadline_before),
            )
        )

    # --- 사업명 키워드 (부분 일치) ---
    if flt.title_keyword:
        must.append(
            models.FieldCondition(
                key="title",
                match=models.MatchText(text=flt.title_keyword),
            )
        )

    # --- 공고 번호 (정확 일치) ---
    if flt.doc_id:
        must.append(
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=flt.doc_id),
            )
        )

    if not must:
        return None

    return models.Filter(must=must)


def merge_filters(
    explicit: MetadataFilter,
    from_query: MetadataFilter,
) -> MetadataFilter:
    """
    명시적 필터와 쿼리 자동 추출 필터 병합
    """
    merged = MetadataFilter()
    for attr in merged.__dataclass_fields__:
        explicit_val = getattr(explicit, attr)
        query_val    = getattr(from_query, attr)
        setattr(merged, attr, explicit_val if explicit_val is not None else query_val)
    return merged


def resolve_filter(
    query: str,
    explicit_filter: Optional[MetadataFilter] = None,
    auto_extract: bool = True,
) -> Optional[models.Filter]:
    """
    최종 Qdrant Filter를 결정하는 통합 진입점
    """
    base = explicit_filter or MetadataFilter()

    if auto_extract:
        from_query = extract_filters_from_query(query)
        merged = merge_filters(explicit=base, from_query=from_query)
    else:
        merged = base

    qdrant_filter = build_qdrant_filter(merged)

    if qdrant_filter:
        _log_active_filters(merged)

    return qdrant_filter


def _log_active_filters(flt: MetadataFilter) -> None:
    active = {k: v for k, v in flt.__dict__.items() if v is not None}
    if active:
        print(f"[필터 적용] {active}")