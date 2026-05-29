import logging
import re
from dataclasses import dataclass, fields
from typing import Optional, List, Tuple, Dict

from qdrant_client import models

# 로거 설정
logger = logging.getLogger(__name__)

# 기관명 추출을 위한 정규식
# 예외적 특수 기관명 (시스템/포털 등 - 사업명과 혼동되지 않도록 고정 명칭 사용)
_PATTERN_EXACT = r"(?:국가과학기술지식정보서비스|KOICA 전자조달|BioIN|나라장터|온비드)"

# 2글자 이상이라 오탐 확률이 낮은 안전한 접미사 (시스템, 서비스 제외)
_SAFE_SUFFIX = (
    r"(?:"
    r"위원회|진흥원|연구원|정보원|평가원|보호원|서비스원|개발원|기술원|교육원|연수원|수련원|의료원"
    r"|연구소|보건소|병원|센터|본부|지사|지부|사무국|사업단|기획단|사무처|산학협력단" # 추가됨
    r"|재단법인|사단법인|공사|공단|재단|조합|협의회|연합회|중앙회|진흥회|체육회|회의소|협회|학회|영화제"
    r"|고등학교|중학교|초등학교|대학교?|대학|학교|교육청"
    r"|테크노파크|박물관|미술관|도서관|과학관|전시관"
    r"|주식회사|유한회사|\([주유]\)|㈜|（[주유]）"
    r"|새마을금고|은행|신협|농협|수협|금고"
    r"|주민센터|행정복지센터|도청|시청|군청|구청"
    r")"
)

_ORG_PREFIX = r"[가-힣a-zA-Z0-9\(\)㈜（）]{1,20}(?:\s+[가-힣a-zA-Z0-9\(\)㈜（）]{1,20}){0,3}"
_PATTERN_SAFE = _ORG_PREFIX + _SAFE_SUFFIX + r"[)\）]?"

# 1글자라 위험한 접미사 (도, 시, 군, 구, 부, 처, 청)
_PATTERN_RISKY = (
    r"(?:[가-힣]{2,4}(?:도|특별시|광역시)\s*)?"
    r"[가-힣]{1,8}(?:도|시|군|구|부|처|청)"
    r"(?=\s|['\"\)\]\）]|에서|이|가|의|는|은|와|과|를|을|나|로|으로|까지|만|$)"
)

# 최종 조립 (우선순위: 정확한 예외명칭 -> 안전한 접미사 패턴 -> 위험한 1글자 패턴)
_ORG_CORE = f"(?:{_PATTERN_EXACT}|{_PATTERN_SAFE}|{_PATTERN_RISKY})"
_PAT_TRIGGER = re.compile(f"({_ORG_CORE})")
_PAT_LABEL = re.compile(f"({_ORG_CORE})")

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
_DATE_PAT = re.compile(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})\s*[일]?")
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


def _is_valid_sub_org(word: str) -> bool:
    """
    쪼개진 어절이 단독으로 유효한 기관명(접미사) 조건을 만족하는지 확인하는 헬퍼 함수
    """
    # 1. 안전한 접미사나 예외 기관명으로 끝나는지 확인
    if re.search(f"(?:{_PATTERN_EXACT}|{_SAFE_SUFFIX}\\)?)$", word):
        return True
    # 2. 1글자 위험 접미사(도, 시, 군, 구, 부, 처, 청) 조건을 만족하는지 확인
    if re.match(r"^[가-힣]{2,8}(?:도|시|군|구|부|처|청)$", word):
        return True
    return False


def _extract_organization(text: str) -> Optional[str]:
    """
    텍스트에서 발주 기관명 추출
    """
    orgs = []
    for pat in (_PAT_TRIGGER, _PAT_LABEL):
        for match in pat.finditer(text):
            candidate = match.group(1).strip()
            if len(candidate) >= 2:
                if candidate not in orgs:
                    orgs.append(candidate)
                
                parts = candidate.split()
                if len(parts) > 1:
                    first_word = parts[0]
                    if len(first_word) >= 2 and _is_valid_sub_org(first_word):
                        if first_word not in orgs:
                            orgs.append(first_word)
                    
                    last_word = parts[-1]
                    if len(last_word) >= 2 and _is_valid_sub_org(last_word):
                        if last_word not in orgs:
                            orgs.append(last_word)
                            
                    if len(parts) > 2:
                        if _is_valid_sub_org(parts[1]):
                            two_words = f"{parts[0]} {parts[1]}"
                            if two_words not in orgs:
                                orgs.append(two_words)

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
            
        # 키워드 앞뒤로 탐색 윈도우(부분 문자열) 생성
        start_idx = max(0, idx - _DATE_SEARCH_WINDOW_PREV)
        end_idx = idx + _DATE_SEARCH_WINDOW_NEXT
        snippet = text[start_idx:end_idx]
        
        match = _DATE_PAT.search(snippet)
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)
            day = match.group(3).zfill(2)
            return f"{year}-{month}-{day}"
            
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
    
    flt.announcement_after = _extract_date(query, ["공고일", "공개일", "공고 이후"])
    flt.bid_deadline_before = _extract_date(query, ["마감", "마감일", "입찰 마감", "기한"])
    flt.bid_start_after = _extract_date(query, ["입찰 시작", "시작일", "접수 시작"])
    
    return flt


def _to_start_of_day(date_str: Optional[str]) -> Optional[str]:
    if date_str and len(date_str) == 10:
        return f"{date_str}T00:00:00Z"
    return date_str


def _to_end_of_day(date_str: Optional[str]) -> Optional[str]:
    if date_str and len(date_str) == 10:
        return f"{date_str}T23:59:59Z"
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

    if flt.bid_start_after:
        must_conditions.append(models.FieldCondition(
            key="bid_start",
            range=models.DatetimeRange(gte=flt.bid_start_after),
        ))

    if flt.bid_deadline_before:
        must_conditions.append(models.FieldCondition(
            key="bid_deadline",
            range=models.DatetimeRange(lte=_to_end_of_day(flt.bid_deadline_before)),
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