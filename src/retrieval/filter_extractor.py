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
    r"|재단법인|재단|공단|공사|협회|센터|위원회|영화제|조합"
    r"|대학교?|대학|의료원|병원|본부|기술원"
    r"|시청|군청|구청|도청"
    r"|[시군구도부처원청사]"
    r")"
)

_ORG_CORE = r"[가-힣a-zA-Z0-9]{1,20}" + _SUFFIX

_PAT_TRIGGER = re.compile(
    r"(?<!\S)"                                          
    r"(" + _ORG_CORE + r")"
    r"(?:\s*(?:에서|이|가|의))?\s*"
    r"(?:[가-힣a-zA-Z0-9()]+\s+){0,5}"
    r"(?:용역|사업|과업|지침|발주|공고|공모|입찰|제안|구축|개선|고도화|재구축|개발|운영)"
)

_PAT_LABEL = re.compile(
    r"(?:발주\s*기관|발주\s*처|발주처|기관명?|수요\s*기관)"
    r"\s*[:\s은는이가]\s*"
    r"(" + _ORG_CORE + r")"                             
)


def _extract_organization(text: str) -> Optional[str]:
    for pat in (_PAT_TRIGGER, _PAT_LABEL):
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) >= 3:
                return candidate
    return None


_BUDGET_UNIT: dict[str, float] = {
    "억":   100_000_000.0,
    "천만": 10_000_000.0,
    "백만": 1_000_000.0,
    "만":   10_000.0,
}

def _parse_budget_string(text: str) -> float:
    total = 0.0
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*([억천백만])", text)
    for num_str, unit_str in matches:
        if num_str and unit_str in _BUDGET_UNIT:
            total += float(num_str) * _BUDGET_UNIT[unit_str]
    return total if total > 0 else 0.0


def _extract_budget(text: str) -> tuple[Optional[float], Optional[float]]:
    budget_min = budget_max = None
    
    # 1개 이상의 숫자+단위 조합을 잡는 패턴 (예: "1억 5천만", "500만")
    money_pat = r"((?:\d+(?:\.\d+)?\s*[억천백만]\s*)+)"

    # A ~ B 
    m = re.search(f"{money_pat}\\s*(?:~|에서|부터|∼)\\s*{money_pat}", text)
    if m:
        v1 = _parse_budget_string(m.group(1))
        v2 = _parse_budget_string(m.group(2))
        if v1 and v2:
            return min(v1, v2), max(v1, v2)

    # 이상 / 초과
    m = re.search(f"{money_pat}\\s*(?:이상|초과|넘는)", text)
    if m:
        val = _parse_budget_string(m.group(1))
        if val: budget_min = val

    # 이하 / 미만 / 이내
    m = re.search(f"{money_pat}\\s*(?:이하|미만|이내)", text)
    if m:
        val = _parse_budget_string(m.group(1))
        if val: budget_max = val

    return budget_min, budget_max


def _extract_date(text: str, keywords: list[str]) -> Optional[str]:
    date_pat = re.compile(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})\s*[일]?")
    for kw in keywords:
        idx = text.find(kw)
        if idx == -1:
            continue
        snippet = text[max(0, idx - 40): idx + 50]
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
    flt.bid_deadline_before        = _extract_date(query, ["마감", "마감일", "입찰 마감", "기한"])
    flt.bid_start_after            = _extract_date(query, ["입찰 시작", "시작일", "접수 시작"])
    return flt


def _to_end_of_day(date_str: Optional[str]) -> Optional[str]:
    if date_str and len(date_str) == 10:  # "YYYY-MM-DD" 형태인 경우
        return f"{date_str} 23:59:59"
    return date_str


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
            range=models.DatetimeRange(
                gte=flt.announcement_after, 
                lte=_to_end_of_day(flt.announcement_before)
            ),
        ))

    if flt.bid_start_after:
        must.append(models.FieldCondition(
            key="bid_start",
            range=models.DatetimeRange(gte=flt.bid_start_after),
        ))

    if flt.bid_deadline_before:
        must.append(models.FieldCondition(
            key="bid_deadline",
            range=models.DatetimeRange(lte=_to_end_of_day(flt.bid_deadline_before)),
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