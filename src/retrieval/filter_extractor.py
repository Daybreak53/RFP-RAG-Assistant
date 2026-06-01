import logging
import re
import csv
import os
import calendar
from dataclasses import dataclass, fields
from typing import Optional, List, Tuple, Dict

from qdrant_client import models

# 로거 설정
logger = logging.getLogger(__name__)

# 기관명 추출을 위한 정규식
def _load_organizations_from_csv(file_path: str) -> List[str]:
    """
    CSV 파일에서 기관명 목록을 읽어와 길이를 기준으로 내림차순 정렬하여 반환합니다.
    (긴 기관명이 부분 문자열 오류 없이 먼저 매칭되도록 처리)
    """
    org_set = set()
    
    if not os.path.exists(file_path):
        logger.warning(f"경고: '{file_path}' 파일을 찾을 수 없습니다. 기관명 필터가 동작하지 않을 수 있습니다.")
        return []
        
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            
            if headers:
                target_idx = None
                for i, header in enumerate(headers):
                    if header.strip() == "발주 기관": 
                        target_idx = i
                        break
                
                if target_idx is None:
                    logger.warning(f"경고: '{file_path}' 파일에 '발주 기관' 컬럼이 없습니다.")
                    return []
                    
                for row in reader:
                    if len(row) > target_idx and row[target_idx].strip():
                        org_set.add(row[target_idx].strip())
                        
    except Exception as e:
        logger.error(f"기관 목록 로드 중 오류 발생: {e}")
        
    return sorted(list(org_set), key=len, reverse=True)


CSV_FILE_PATH = "data/data_list.csv" 
_KNOWN_ORGS = _load_organizations_from_csv(CSV_FILE_PATH)

# 예산 단위 변환 맵
_BUDGET_UNIT: Dict[str, float] = {
    "억":   100_000_000.0,
    "천만": 10_000_000.0,
    "천":   10_000_000.0,
    "백만": 1_000_000.0,
    "백":   1_000_000.0,
    "만":   10_000.0,
}

# 예산 및 날짜 관련 정규식/상수
_MONEY_PAT = r"((?:\d+(?:\.\d+)?\s*(?:억|천만|백만|천|백|만)\s*)+)"
_DATE_PAT = re.compile(r"(\d{4})\s*(?:년|[-./])\s*(?:(\d{1,2})\s*(?:월|[-./])\s*)?(?:(\d{1,2})\s*일?)?")
_DATE_SEARCH_WINDOW_PREV = 40  # 키워드 앞 탐색 글자 수
_DATE_SEARCH_WINDOW_NEXT = 50  # 키워드 뒤 탐색 글자 수


@dataclass
class MetadataFilter:
    """
    검색에 사용할 메타데이터 필터 조건 집합
    """
    organization: Optional[List[str]] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    announcement_after: Optional[str] = None
    announcement_before: Optional[str] = None
    bid_start_after: Optional[str] = None
    bid_start_before: Optional[str] = None
    bid_deadline_after: Optional[str] = None
    bid_deadline_before: Optional[str] = None
    title_keyword: Optional[str] = None
    doc_id: Optional[str] = None

    def is_empty(self) -> bool:
        """
        모든 필터 조건이 비어있는지(None) 확인
        """
        return all(getattr(self, field.name) is None for field in fields(self))

    def merge_with(self, other: "MetadataFilter") -> "MetadataFilter":
        """
        현재 필터에 다른 필터(other) 병합
        """
        merged = MetadataFilter()
        for field in fields(self):
            if field.name == "organization":
                self_orgs = getattr(self, field.name) or []
                other_orgs = getattr(other, field.name) or []
                merged_orgs = list(dict.fromkeys(self_orgs + other_orgs))
                setattr(merged, field.name, merged_orgs if merged_orgs else None)
            else:
                self_val = getattr(self, field.name)
                other_val = getattr(other, field.name)
                setattr(merged, field.name, self_val if self_val is not None else other_val)
        return merged


def _extract_organization(text: str) -> Optional[List[str]]:
    """
    텍스트에서 발주 기관명 추출
    """
    if not _KNOWN_ORGS:
        return None
        
    orgs = []
    for org in _KNOWN_ORGS:
        if org in text:
            is_subpart = any(org in found_org for found_org in orgs)
            if not is_subpart:
                orgs.append(org)
                
    return orgs if orgs else None


def _parse_budget_string(text: str) -> float:
    """
    '1억 5천만' 등의 문자열을 숫자(float) 파싱
    """
    total = 0.0
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(억|천만|백만|천|백|만)", text)
    for num_str, unit_str in matches:
        if num_str and unit_str in _BUDGET_UNIT:
            total += float(num_str) * _BUDGET_UNIT[unit_str]
    return total if total > 0 else 0.0


def _extract_budget(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    텍스트에서 예산의 최소/최대 조건 추출
    """
    budget_min, budget_max = None, None

    # 범위 (A ~ B, A에서 B까지 등)
    range_match = re.search(f"{_MONEY_PAT}\\s*(?:~|에서|부터|∼)\\s*{_MONEY_PAT}", text)
    if range_match:
        val1 = _parse_budget_string(range_match.group(1))
        val2 = _parse_budget_string(range_match.group(2))
        if val1 and val2:
            return min(val1, val2), max(val1, val2)

    # 하한선 (이상 / 초과)
    min_match = re.search(f"{_MONEY_PAT}\\s*(?:이상|초과|넘는)", text)
    if min_match:
        val = _parse_budget_string(min_match.group(1))
        if val:
            budget_min = val

    # 상한선 (이하 / 미만 / 이내)
    max_match = re.search(f"{_MONEY_PAT}\\s*(?:이하|미만|이내)", text)
    if max_match:
        val = _parse_budget_string(max_match.group(1))
        if val:
            budget_max = val

    return budget_min, budget_max


def _extract_date(text: str, keywords: List[str]) -> Optional[str]:
    """
    특정 키워드 근처에서 날짜(YYYY-MM-DD) 추출
    """
    for keyword in keywords:
        idx = text.find(keyword)
        if idx == -1:
            continue
            
        start_idx = max(0, idx - _DATE_SEARCH_WINDOW_PREV)
        end_idx = idx + _DATE_SEARCH_WINDOW_NEXT
        snippet = text[start_idx:end_idx]
        
        match = _DATE_PAT.search(snippet)
        if match:
            year = match.group(1)
            month = match.group(2)
            day = match.group(3)

            if month and day:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            elif month:
                return f"{year}-{month.zfill(2)}"
            else:
                return f"{year}"
                
    return None


def extract_filters_from_query(query: str) -> MetadataFilter:
    """
    자연어 질의에서 메타데이터 필터 조건들 자동 추출
    """
    flt = MetadataFilter()
    flt.organization = _extract_organization(query)
    
    budget_min, budget_max = _extract_budget(query)
    flt.budget_min = budget_min
    flt.budget_max = budget_max
    
    # 공고일
    announce_date = _extract_date(query, ["공고일", "공개일"])
    if announce_date:
        if "이후" in query:
            flt.announcement_after = announce_date
        elif "이전" in query or "까지" in query:
            flt.announcement_before = announce_date
        else:
            flt.announcement_after = announce_date
            flt.announcement_before = announce_date

    # 입찰 시작일
    bid_start = _extract_date(query, ["입찰 시작", "시작일", "접수 시작"])
    if bid_start:
        if "이후" in query:
            flt.bid_start_after = bid_start
        elif "이전" in query or "까지" in query:
            flt.bid_start_before = bid_start
        else:
            flt.bid_start_after = bid_start
            flt.bid_start_before = bid_start
    
    # 입찰 마감일
    bid_deadline = _extract_date(query, ["마감", "마감일", "입찰 마감", "기한"])
    if bid_deadline:
        if "이후" in query:
            flt.bid_deadline_after = bid_deadline
        elif "이전" in query or "까지" in query:
            flt.bid_deadline_before = bid_deadline
        else:
            flt.bid_deadline_after = bid_deadline
            flt.bid_deadline_before = bid_deadline
            
    return flt


def _to_start_of_day(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    parts = date_str.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-01-01T00:00:00Z"
    elif len(parts) == 2:
        return f"{parts[0]}-{parts[1]}-01T00:00:00Z"
    elif len(parts) == 3:
        return f"{parts[0]}-{parts[1]}-{parts[2]}T00:00:00Z"
    return date_str


def _to_end_of_day(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    parts = date_str.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-12-31T23:59:59Z"
    elif len(parts) == 2:
        last_day = calendar.monthrange(int(parts[0]), int(parts[1]))[1]
        return f"{parts[0]}-{parts[1]}-{last_day}T23:59:59Z"
    elif len(parts) == 3:
        return f"{parts[0]}-{parts[1]}-{parts[2]}T23:59:59Z"
    return date_str


def build_qdrant_filter(flt: MetadataFilter) -> Optional[models.Filter]:
    """
    MetadataFilter 객체를 Qdrant 검색용 models.Filter로 변환
    """
    if flt is None or flt.is_empty():
        return None

    must_conditions: List[models.Condition] = []

    if flt.organization:
        org_conditions = [
            models.FieldCondition(
                key="organization",
                match=models.MatchText(text=org)
            )
            for org in flt.organization
        ]

        must_conditions.append(
            models.Filter(should=org_conditions)
        )

    if flt.budget_min is not None or flt.budget_max is not None:
        must_conditions.append(models.FieldCondition(
            key="budget",
            range=models.Range(gte=flt.budget_min, lte=flt.budget_max),
        ))

    if flt.announcement_after or flt.announcement_before:
        must_conditions.append(models.FieldCondition(
            key="announcement_date",
            range=models.DatetimeRange(
                gte=_to_start_of_day(flt.announcement_after), 
                lte=_to_end_of_day(flt.announcement_before)
            ),
        ))

    if flt.bid_start_after or flt.bid_start_before:
        must_conditions.append(models.FieldCondition(
            key="bid_start",
            range=models.DatetimeRange(
                gte=_to_start_of_day(flt.bid_start_after), 
                lte=_to_end_of_day(flt.bid_start_before)
            ),
        ))

    if flt.bid_deadline_after or flt.bid_deadline_before:
        must_conditions.append(models.FieldCondition(
            key="bid_deadline",
            range=models.DatetimeRange(
                gte=_to_start_of_day(flt.bid_deadline_after),
                lte=_to_end_of_day(flt.bid_deadline_before)
            ),
        ))

    if flt.title_keyword:
        must_conditions.append(models.FieldCondition(
            key="title",
            match=models.MatchText(text=flt.title_keyword),
        ))

    if flt.doc_id:
        must_conditions.append(models.FieldCondition(
            key="doc_id",
            match=models.MatchValue(value=str(flt.doc_id)),
        ))

    return models.Filter(must=must_conditions) if must_conditions else None


def resolve_filter(
    query: str,
    explicit_filter: Optional[MetadataFilter] = None,
    auto_extract: bool = True,
    query_type: Optional[str] = None,
) -> Optional[models.Filter]:
    """
    명시적(config 기반) 필터와 자연어 질의 기반 자동 추출 필터를 병합하여 
    최종 Qdrant 필터 객체 반환
    """
    base_filter = explicit_filter or MetadataFilter()
    
    if auto_extract:
        extracted_filter = extract_filters_from_query(query)

        # 명시적 필터(base)가 자동 추출 필터(extracted)보다 우선순위를 가짐
        merged_filter = base_filter.merge_with(extracted_filter)
    else:
        merged_filter = base_filter

    qdrant_filter = build_qdrant_filter(merged_filter)
    
    if qdrant_filter:
        # 적용된 필터 조건만 추출하여 로깅
        active_filters = {
            field.name: getattr(merged_filter, field.name)
            for field in fields(merged_filter)
            if getattr(merged_filter, field.name) is not None
        }
        logger.info(f"검색 필터 적용됨: {active_filters}")

    return qdrant_filter