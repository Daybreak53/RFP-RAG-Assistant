from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from qdrant_client import models


# 필터 조건 데이터 클래스
@dataclass
class MetadataFilter:
    """검색에 사용할 메타데이터 필터 조건 집합"""
    organization:        Optional[str]   = None
    budget_min:          Optional[float] = None
    budget_max:          Optional[float] = None
    announcement_after:  Optional[str]   = None
    announcement_before: Optional[str]   = None
    bid_start_after:     Optional[str]   = None
    bid_deadline_before: Optional[str]   = None
    title_keyword:       Optional[str]   = None
    doc_id:              Optional[str]   = None

    def is_empty(self) -> bool:
        return all(v is None for v in self.__dict__.values())
    

# 기관 suffix
_SUFFIX = (
    r"(?:"
    r"특별자치도|특별자치시|특별시|광역시"
    r"|행정안전부|과학기술정보통신부|기획재정부|국토교통부|보건복지부"
    r"|교육부|환경부|고용노동부|산업통상자원부|문화체육관광부"
    r"|진흥원|연구원|연구소|교육청|경찰청|소방청|진흥회"
    r"|재단법인|재단|공단|공사|협회|센터"
    r"|시청|군청|구청|도청"
    r"|[시군구도부처원청]"
    r")"
)

# 기관명 전체 패턴: 순수 한글 2~20자 + suffix
# (?<!\S) 는 직전이 공백·줄시작임을 보장 → 숫자나 단위 바로 뒤 매칭 방지
_ORG_CORE = r"[가-힣]{2,20}" + _SUFFIX

_PAT_TRIGGER = re.compile(
    r"(?<!\S)"                                          # 앞이 공백/시작
    r"(" + _ORG_CORE + r")"                             # 기관명 캡처
    r"\s*(?:에서|이|가|의)?\s*"
    r"(?:발주|공고|공모|입찰|제안)"                       # 트리거 단어
)

_PAT_LABEL = re.compile(
    r"(?:발주\s*기관|발주\s*처|발주처|기관명?|수요\s*기관)"
    r"\s*[:\s은는이가]\s*"
    r"(" + _ORG_CORE + r")"                             # 기관명 캡처
)


def _extract_organization(text: str) -> Optional[str]:
    for pat in (_PAT_TRIGGER, _PAT_LABEL):
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) >= 3:
                return candidate
    return None


# 예산 추출
_BUDGET_UNIT: dict[str, float] = {
    "억":   10_000,
    "천만": 1_000,
    "백만": 100,
    "만":   1,
}


def _unit_to_manwon(unit_str: str) -> float:
    for key, val in _BUDGET_UNIT.items():
        if key in unit_str:
            return val
    return 1.0


def _extract_budget(text: str) -> tuple[Optional[float], Optional[float]]:
    budget_min = budget_max = None

    m = re.search(
        r"(\d+(?:\.\d+)?)\s*([억천백만]+)\s*(?:~|에서|부터|∼)\s*(\d+(?:\.\d+)?)\s*([억천백만]+)",
        text
    )
    if m:
        v1 = float(m.group(1)) * _unit_to_manwon(m.group(2))
        v2 = float(m.group(3)) * _unit_to_manwon(m.group(4))
        return min(v1, v2), max(v1, v2)

    m = re.search(r"(\d+(?:\.\d+)?)\s*([억천백만]+)\s*(?:이상|초과|넘는)", text)
    if m:
        budget_min = float(m.group(1)) * _unit_to_manwon(m.group(2))

    m = re.search(r"(\d+(?:\.\d+)?)\s*([억천백만]+)\s*(?:이하|미만|이내)", text)
    if m:
        budget_max = float(m.group(1)) * _unit_to_manwon(m.group(2))

    return budget_min, budget_max


# 날짜 추출
def _extract_date(text: str, keywords: list[str]) -> Optional[str]:
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


# 통합 자동 추출
def extract_filters_from_query(query: str) -> MetadataFilter:
    flt = MetadataFilter()
    flt.organization               = _extract_organization(query)
    flt.budget_min, flt.budget_max = _extract_budget(query)
    flt.announcement_after         = _extract_date(query, ["공고일", "공개일", "공고 이후"])
    flt.bid_deadline_before        = _extract_date(query, ["마감", "마감일", "입찰 마감"])
    flt.bid_start_after            = _extract_date(query, ["입찰 시작", "시작일", "접수 시작"])
    return flt


# MetadataFilter → Qdrant Filter 변환
def build_qdrant_filter(flt: MetadataFilter) -> Optional[models.Filter]:
    if flt is None or flt.is_empty():
        return None

    must: list[models.Condition] = []

    if flt.organization:
        must.append(models.FieldCondition(
            key="organization",
            match=models.MatchText(text=flt.organization),
        ))

    if flt.budget_min is not None or flt.budget_max is not None:
        must.append(models.FieldCondition(
            key="budget",
            range=models.Range(gte=flt.budget_min, lte=flt.budget_max),
        ))

    if flt.announcement_after or flt.announcement_before:
        must.append(models.FieldCondition(
            key="announcement_date",
            range=models.Range(gte=flt.announcement_after, lte=flt.announcement_before),
        ))

    if flt.bid_start_after:
        must.append(models.FieldCondition(
            key="bid_start",
            range=models.Range(gte=flt.bid_start_after),
        ))

    if flt.bid_deadline_before:
        must.append(models.FieldCondition(
            key="bid_deadline",
            range=models.Range(lte=flt.bid_deadline_before),
        ))

    if flt.title_keyword:
        must.append(models.FieldCondition(
            key="title",
            match=models.MatchText(text=flt.title_keyword),
        ))

    if flt.doc_id:
        must.append(models.FieldCondition(
            key="doc_id",
            match=models.MatchValue(value=flt.doc_id),
        ))

    return models.Filter(must=must) if must else None


# 병합 및 최종 resolve
def merge_filters(explicit: MetadataFilter, from_query: MetadataFilter) -> MetadataFilter:
    merged = MetadataFilter()
    for attr in merged.__dataclass_fields__:
        setattr(merged, attr,
                getattr(explicit, attr) if getattr(explicit, attr) is not None
                else getattr(from_query, attr))
    return merged


def resolve_filter(
    query: str,
    explicit_filter: Optional[MetadataFilter] = None,
    auto_extract: bool = True,
) -> Optional[models.Filter]:
    base = explicit_filter or MetadataFilter()
    merged = merge_filters(base, extract_filters_from_query(query)) if auto_extract else base

    qdrant_filter = build_qdrant_filter(merged)
    if qdrant_filter:
        active = {k: v for k, v in merged.__dict__.items() if v is not None}
        print(f"[필터 적용] {active}")

    return qdrant_filter