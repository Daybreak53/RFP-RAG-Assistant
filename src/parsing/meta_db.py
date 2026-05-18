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


def normalize_filename(name):
    if not isinstance(name, str):
        return ""

    name = unicodedata.normalize("NFC", name)
    name = name.replace("\\", "/")
    name = os.path.basename(name)

    name = re.sub(r"\.(pdf|hwp|hwpx)$", "", name, flags=re.IGNORECASE)
    name = name.lower()
    name = re.sub(r"\s+", "", name)

    for ch in ["_", "-", "&", "(", ")", "[", "]", "{", "}", ".", ",", "·", "ㆍ"]:
        name = name.replace(ch, "")

    return name


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def find_best_csv_match(actual_file, csv_file_names):
    actual_norm = normalize_filename(actual_file)

    best_score = 0
    best_csv_file = None
    best_match_type = None

    for csv_file in csv_file_names:
        csv_norm = normalize_filename(csv_file)

        if actual_norm == csv_norm:
            return csv_file, 1.0, "exact"

        if actual_norm in csv_norm or csv_norm in actual_norm:
            score = 0.95
            match_type = "contains"
        else:
            score = similarity(actual_norm, csv_norm)
            match_type = "similarity"

        if score > best_score:
            best_score = score
            best_csv_file = csv_file
            best_match_type = match_type

    return best_csv_file, best_score, best_match_type


def load_metadata_db(csv_path: str, data_dir: str, threshold: float = 0.55):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data 폴더가 없습니다: {data_dir}")

    metadata_df = pd.read_csv(csv_path)

    actual_files = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
    ]

    csv_file_names = metadata_df["파일명"].astype(str).tolist()

    metadata_map = {}
    match_results = []

    for actual_file in actual_files:
        best_csv_file, score, match_type = find_best_csv_match(
            actual_file,
            csv_file_names
        )

        matched = score >= threshold

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
                "file_name": actual_file,
                "file_type": row["파일형식"],
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
