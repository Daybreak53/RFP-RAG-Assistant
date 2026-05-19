"""
CSV 기반 메타데이터 로드 및 파일명 매칭 모듈입니다.

현재는 CSV를 사용하지만, 나중에 실제 DB로 바꿀 경우
load_metadata_db() 함수 내부만 수정하면 됩니다.
"""

import os
import re
import unicodedata
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote, urlparse


SOURCE_EXTENSIONS = {".hwp", ".pdf"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def normalize_source_filename(name):
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
    return os.path.basename(name).strip()


def normalize_filename(name):
    filename = normalize_source_filename(name)
    if not filename:
        return ""

    filename = unicodedata.normalize("NFC", filename)
    filename = re.sub(r"\s+", " ", filename).strip()
    filename = re.sub(r"\s+(?=\.[^.]+$)", "", filename)
    return filename.casefold()


def _source_extension(name):
    return os.path.splitext(normalize_source_filename(name))[1].casefold()


def _without_copy_suffix_key(name):
    filename = normalize_filename(name)
    stem, ext = os.path.splitext(filename)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return f"{stem}{ext}"


def _stem_key(name):
    stem, _ = os.path.splitext(normalize_filename(name))
    return re.sub(r"\s*\(\d+\)$", "", stem)


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def resolve_source_filename(name, data_dir=None):
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

    requested_key = normalize_filename(file_name)
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
            (similarity(requested_key, normalize_filename(candidate)), candidate)
            for candidate in candidates
        ),
        reverse=True,
    )
    if ranked and ranked[0][0] >= 0.8:
        runner_up = ranked[1][0] if len(ranked) > 1 else 0
        if ranked[0][0] - runner_up >= 0.05:
            return ranked[0][1]

    return file_name


def find_best_csv_match(actual_file, csv_file_names):
    actual_norm = normalize_filename(actual_file)
    actual_copyless_norm = _without_copy_suffix_key(actual_file)
    actual_ext = _source_extension(actual_file)

    best_score = 0
    best_csv_file = None
    best_match_type = None

    for csv_file in csv_file_names:
        if actual_ext and _source_extension(csv_file) != actual_ext:
            continue

        csv_norm = normalize_filename(csv_file)
        csv_copyless_norm = _without_copy_suffix_key(csv_file)

        if actual_norm == csv_norm:
            return csv_file, 1.0, "exact"

        if actual_copyless_norm == csv_norm or actual_norm == csv_copyless_norm:
            score = 0.99
            match_type = "copy_suffix"
        else:
            score = similarity(actual_norm, csv_norm)
            match_type = "similarity"

        if score > best_score:
            best_score = score
            best_csv_file = csv_file
            best_match_type = match_type

    return best_csv_file, best_score, best_match_type


def load_metadata_db(csv_path: str, data_dir: str, threshold: float = 0.55):
    if threshold is None:
        threshold = 0.55

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data 폴더가 없습니다: {data_dir}")

    metadata_df = pd.read_csv(csv_path)

    actual_files = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
        and _source_extension(f) in SOURCE_EXTENSIONS
    ]
    actual_files.sort(key=normalize_filename)

    csv_file_names = metadata_df["파일명"].astype(str).tolist()

    metadata_map = {}
    match_results = []

    for actual_file in actual_files:
        best_csv_file, score, match_type = find_best_csv_match(
            actual_file,
            csv_file_names
        )

        match_threshold = threshold
        if match_type == "similarity":
            match_threshold = max(threshold, 0.9)

        matched = score >= match_threshold

        if matched:
            row = metadata_df[
                metadata_df["파일명"].astype(str) == best_csv_file
            ].iloc[0]

            actual_key = normalize_filename(actual_file)

            metadata_map[actual_key] = {
                "doc_id": str(row["공고 번호"]).strip(),
                "title": row["사업명"],
                "organization": row["발주 기관"],
                "budget": row["사업 금액"],
                "announcement_date": row["공개 일자"],
                "bid_start": row["입찰 참여 시작일"],
                "bid_deadline": row["입찰 참여 마감일"],
                "section_title": row["사업 요약"],
                "file_name": normalize_source_filename(actual_file),
                "file_type": _source_extension(actual_file).lstrip("."),
            }

        match_results.append({
            "actual_file": actual_file,
            "best_csv_file": best_csv_file,
            "score": score,
            "match_type": match_type,
            "matched": matched,
        })

    match_df = pd.DataFrame(match_results)
    return metadata_map, match_df
