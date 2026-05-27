import logging
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import pandas as pd

# 로거 설정
logger = logging.getLogger(__name__)

SOURCE_EXTENSIONS = {".hwp", ".pdf", ".hwpx"}
PROJECT_ROOT     = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

_EXT_PATTERN           = re.compile(r"\.(pdf|hwp|hwpx)$", flags=re.IGNORECASE)
_SPECIAL_CHARS_PATTERN = re.compile(r"[\s_\-&()\[\]{}.,·ㆍ]")


def normalize_source_filename(name: Any) -> str:
    """
    URL/경로 형태의 입력에서 순수한 원본 파일명(확장자 포함)을 추출
    """
    if not isinstance(name, str):
        return ""

    name = name.strip()
    if not name:
        return ""

    parsed = urlparse(name)
    if parsed.scheme in {"http", "https", "file"} and parsed.path:
        name = parsed.path

    name = unquote(name)
    name = name.replace("\\", "/")
    return Path(name).name.strip()


def normalize_filename(name: str) -> str:
    """
    비교를 위해 파일명 정규화 (확장자 제거, 특수문자 제거, 소문자화)
    """
    filename = normalize_source_filename(name)
    if not filename:
        return ""

    # Mac/Windows 간 자음/모음 분리(NFD/NFC) 문제 해결
    filename = unicodedata.normalize("NFC", filename)

    # 확장자 제거 및 소문자 변환
    filename = _EXT_PATTERN.sub("", filename).lower()

    # 정규식을 이용한 특수기호 및 공백 일괄 제거
    return _SPECIAL_CHARS_PATTERN.sub("", filename).casefold()


def _source_extension(name: str) -> str:
    return Path(normalize_source_filename(name)).suffix.casefold()


def _without_copy_suffix_key(name: str) -> str:
    stem, ext = os.path.splitext(normalize_filename(name))
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return f"{stem}{ext}"


def _stem_key(name: str) -> str:
    stem, _ = os.path.splitext(normalize_filename(name))
    return re.sub(r"\s*\(\d+\)$", "", stem)


def calculate_similarity(a: str, b: str) -> float:
    """두 문자열 간의 유사도 점수(0.0 ~ 1.0) 계산"""
    return SequenceMatcher(None, a, b).ratio()


def resolve_source_filename(name: Any, data_dir: Optional[Union[str, Path]] = None) -> str:
    """
    파일명 후보를 실제 data 디렉토리의 파일명으로 매칭 (정확/copy suffix/유사도 순)
    """
    file_name = normalize_source_filename(name)
    if _source_extension(file_name) not in SOURCE_EXTENSIONS:
        return file_name

    source_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if not source_dir.exists():
        return file_name

    candidates = [
        p.name for p in source_dir.iterdir()
        if p.is_file() and _source_extension(p.name) == _source_extension(file_name)
    ]
    if not candidates:
        return file_name

    requested_key          = normalize_filename(file_name)
    requested_copyless_key = _without_copy_suffix_key(file_name)
    for candidate in candidates:
        candidate_key = normalize_filename(candidate)
        if requested_key == candidate_key:
            return candidate
        if requested_copyless_key == _without_copy_suffix_key(candidate):
            return candidate

    requested_stem = _stem_key(file_name)
    prefix_matches = [
        candidate for candidate in candidates
        if len(requested_stem) >= 12 and _stem_key(candidate).startswith(requested_stem)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    ranked = sorted(
        (
            (calculate_similarity(requested_key, normalize_filename(candidate)), candidate)
            for candidate in candidates
        ),
        reverse=True,
    )
    if ranked and ranked[0][0] >= 0.8:
        runner_up = ranked[1][0] if len(ranked) > 1 else 0
        if ranked[0][0] - runner_up >= 0.05:
            return ranked[0][1]

    return file_name


def find_best_csv_match(
    actual_file: str,
    csv_file_names: List[str],
) -> Tuple[Optional[str], float, str]:
    """
    실제 파일명과 가장 유사한 CSV 메타데이터 파일명 찾기
    """
    actual_norm          = normalize_filename(actual_file)
    actual_copyless_norm = _without_copy_suffix_key(actual_file)
    actual_ext           = _source_extension(actual_file)
    if not actual_norm:
        return None, 0.0, "none"

    best_score      = 0.0
    best_csv_file   = None
    best_match_type = "none"

    for csv_file in csv_file_names:
        if actual_ext and _source_extension(csv_file) != actual_ext:
            continue

        csv_norm = normalize_filename(csv_file)
        if not csv_norm:
            continue

        csv_copyless_norm = _without_copy_suffix_key(csv_file)

        # 1. 완전 일치
        if actual_norm == csv_norm:
            return csv_file, 1.0, "exact"

        # 2. copy suffix 변형 일치 ("foo (1).pdf" ↔ "foo.pdf")
        if actual_copyless_norm == csv_norm or actual_norm == csv_copyless_norm:
            score = 0.99
            match_type = "copy_suffix"
        # 3. 부분 일치 (포함 관계)
        elif actual_norm in csv_norm or csv_norm in actual_norm:
            score = 0.95
            match_type = "contains"
        else:
            # 4. 텍스트 유사도
            score = calculate_similarity(actual_norm, csv_norm)
            match_type = "similarity"

        if score > best_score:
            best_score = score
            best_csv_file = csv_file
            best_match_type = match_type

    return best_csv_file, best_score, best_match_type


def load_metadata_db(
    csv_path: Union[str, Path],
    data_dir: Union[str, Path],
    threshold: float = 0.55,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    CSV 메타데이터를 로드하고, 데이터 디렉토리 내의 실제 파일들과 매핑
    """
    if threshold is None:
        threshold = 0.55

    csv_path = Path(csv_path)
    data_dir = Path(data_dir)

    if not csv_path.is_file():
        logger.error(f"메타데이터 CSV 파일을 찾을 수 없습니다: {csv_path}")
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    if not data_dir.is_dir():
        logger.error(f"데이터 폴더를 찾을 수 없습니다: {data_dir}")
        raise FileNotFoundError(f"data 폴더가 없습니다: {data_dir}")

    logger.info(f"메타데이터 로드 및 파일 매칭 시작 (임계값: {threshold})")

    try:
        metadata_df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"CSV 파일 파싱 중 오류 발생: {e}", exc_info=True)
        raise

    # 필수 컬럼 체크
    if "파일명" not in metadata_df.columns:
        raise ValueError("CSV 파일에 '파일명' 컬럼이 존재하지 않습니다.")

    # 실제 파일 목록 및 CSV 대상 파일명 추출 (지원 확장자만, 정규화된 이름 순)
    actual_files = sorted(
        (f.name for f in data_dir.iterdir()
         if f.is_file() and _source_extension(f.name) in SOURCE_EXTENSIONS),
        key=normalize_filename,
    )
    csv_file_names = metadata_df["파일명"].dropna().astype(str).tolist()

    metadata_map = {}
    match_results = []

    for actual_file in actual_files:
        best_csv_file, score, match_type = find_best_csv_match(actual_file, csv_file_names)

        # similarity 매칭은 더 엄격한 임계값 적용
        match_threshold = max(threshold, 0.9) if match_type == "similarity" else threshold
        matched = score >= match_threshold

        if matched and best_csv_file:
            row = metadata_df[metadata_df["파일명"].astype(str) == best_csv_file].iloc[0]
            actual_key = normalize_filename(actual_file)

            raw_budget = row.get("사업 금액")
            try:
                budget_value = float(str(raw_budget).replace(",", "").replace("원", "").strip())
            except (ValueError, TypeError):
                budget_value = None

            metadata_map[actual_key] = {
                "doc_id":            str(row.get("공고 번호", "")).strip(),
                "title":             row.get("사업명"),
                "organization":      row.get("발주 기관"),
                "budget":            budget_value,
                "announcement_date": row.get("공개 일자"),
                "bid_start":         row.get("입찰 참여 시작일"),
                "bid_deadline":      row.get("입찰 참여 마감일"),
                "section_title":     row.get("사업 요약"),
                "file_name":         normalize_source_filename(actual_file),
                "file_type":         row.get("파일형식") or _source_extension(actual_file).lstrip("."),
            }

        match_results.append({
            "actual_file": actual_file,
            "best_csv_file": best_csv_file,
            "score": score,
            "match_type": match_type,
            "matched": matched,
        })

    match_df = pd.DataFrame(match_results)

    # 매칭 결과 통계 로깅
    success_count = match_df["matched"].sum()
    total_count = len(match_df)
    match_ratio = (success_count / total_count * 100) if total_count > 0 else 0
    
    logger.info(f"문서 메타데이터 매칭 완료: {success_count}/{total_count} 성공 (매칭률: {match_ratio:.1f}%)")
    
    if success_count < total_count:
        logger.warning(f"매칭 실패 건수: {total_count - success_count}건. (threshold={threshold})")

    return metadata_map, match_df