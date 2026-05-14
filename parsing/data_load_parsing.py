"""
RAG JSON 생성 최종 스크립트

처리 흐름:
1. parsing/data 폴더의 PDF/HWP 파일 로드
2. parsing/data_list.csv 메타데이터 매칭
3. 문서 chunk 분할
4. RAG 저장 형식으로 변환
5. parsing/rag_data.json 저장

필요 패키지:
pip install pandas pypdf langchain==0.1.20 langchain-community==0.0.38 langchain-core==0.1.52 langchain-text-splitters==0.0.1 langchain-teddynote
"""

import os
import re
import json
import unicodedata
import pandas as pd

from difflib import SequenceMatcher

from langchain_community.document_loaders import PyPDFLoader
from langchain_teddynote.document_loaders import HWPLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. 경로 설정
# ============================================================

# 이 파일(data_load_parsing.py)이 있는 폴더를 기준으로 경로를 잡습니다.
# 따라서 프로젝트 루트에서 실행해도, parsing 폴더 안에서 실행해도 동일하게 동작합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(BASE_DIR, "data_list.csv")

RAG_JSON_PATH = os.path.join(BASE_DIR, "rag_data.json")
RAG_SAMPLE_JSON_PATH = os.path.join(BASE_DIR, "rag_data_sample.json")
MATCH_RESULT_CSV_PATH = os.path.join(BASE_DIR, "metadata_match_result.csv")
CHUNK_EXPERIMENT_CSV_PATH = os.path.join(BASE_DIR, "chunk_size_experiment_results.csv")

SELECTED_CHUNK_SIZE = 500
MATCH_THRESHOLD = 0.55


# ============================================================
# 2. 파일명 정규화 및 매칭 함수
# ============================================================

def normalize_filename(name):
    """
    파일명 매칭을 위한 정규화 함수입니다.

    처리:
    - 유니코드 정규화
    - 경로 제거
    - 확장자 제거
    - 소문자 변환
    - 공백 제거
    - 주요 특수문자 제거
    - 한글은 유지
    """

    if not isinstance(name, str):
        return ""

    name = unicodedata.normalize("NFC", name)

    name = name.replace("\\", "/")
    name = os.path.basename(name)

    name = re.sub(
        r"\.(pdf|hwp|hwpx)$",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = name.lower()
    name = re.sub(r"\s+", "", name)

    remove_chars = [
        "_", "-", "&", "(", ")", "[", "]",
        "{", "}", ".", ",", "·", "ㆍ"
    ]

    for ch in remove_chars:
        name = name.replace(ch, "")

    return name


def similarity(a, b):
    """두 문자열의 유사도를 계산합니다."""
    return SequenceMatcher(None, a, b).ratio()


def find_best_csv_match(actual_file, csv_file_names):
    """
    실제 파일명과 CSV 파일명 목록 중 가장 유사한 항목을 찾습니다.

    매칭 우선순위:
    1. 완전 일치
    2. 포함 관계
    3. 유사도
    """

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


# ============================================================
# 3. CSV 메타데이터 매칭
# ============================================================

def load_metadata_csv_matched(
    csv_path=CSV_PATH,
    data_dir=DATA_DIR,
    threshold=MATCH_THRESHOLD
):
    """
    실제 data 폴더 파일명과 CSV 파일명을 매칭하여 metadata_map을 생성합니다.
    """

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
            "matched": matched
        })

    match_df = pd.DataFrame(match_results)

    return metadata_map, match_df


# ============================================================
# 4. PDF/HWP 문서 로드
# ============================================================

def load_documents(data_dir=DATA_DIR):
    """
    data 폴더의 PDF/HWP 문서를 모두 로드합니다.
    """

    documents = []

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)

        if not os.path.isfile(file_path):
            continue

        if file.lower().endswith(".pdf"):
            print("PDF 로드:", file)

            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

            except Exception as e:
                print("PDF 로드 실패:", file)
                print(e)

        elif file.lower().endswith(".hwp"):
            print("HWP 로드:", file)

            try:
                loader = HWPLoader(file_path)
                documents.extend(loader.load())

            except Exception as e:
                print("HWP 로드 실패:", file)
                print(e)

        else:
            print("제외:", file)

    print("\n전체 로드된 Document 수:", len(documents))

    return documents


# ============================================================
# 5. Chunk 생성 및 실험
# ============================================================

def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    """
    문서를 chunk로 분할합니다.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(documents)


def chunk_size_experiment(
    documents,
    start=100,
    end=1000,
    step=100,
    overlap_ratio=0.1
):
    """
    chunk_size를 100~1000까지 100 단위로 실험합니다.
    """

    results = []

    for chunk_size in range(start, end + 1, step):
        chunk_overlap = int(chunk_size * overlap_ratio)

        chunks = create_chunks(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        results.append({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": len(chunks)
        })

        print("=" * 50)
        print("chunk_size:", chunk_size)
        print("chunk_overlap:", chunk_overlap)
        print("num_chunks:", len(chunks))

    return pd.DataFrame(results)


# ============================================================
# 6. RAG JSON 변환
# ============================================================

def convert_chunks_to_rag_format(chunks, metadata_map=None):
    """
    chunk 리스트를 RAG 저장용 JSON 형식으로 변환합니다.
    """

    rag_data = []
    chunk_count_map = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        file_name = os.path.basename(source)
        file_key = normalize_filename(file_name)

        csv_meta = metadata_map.get(file_key, {}) if metadata_map else {}

        doc_id = csv_meta.get("doc_id", os.path.splitext(file_name)[0])

        if doc_id not in chunk_count_map:
            chunk_count_map[doc_id] = 0

        chunk_count_map[doc_id] += 1
        chunk_number = chunk_count_map[doc_id]

        chunk_id = f"{doc_id}_{chunk_number:04d}"

        item = {
            "id": chunk_id,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": csv_meta.get("title", os.path.splitext(file_name)[0]),
                "organization": csv_meta.get("organization", None),
                "budget": csv_meta.get("budget", None),
                "announcement_date": csv_meta.get("announcement_date", None),
                "bid_start": csv_meta.get("bid_start", None),
                "bid_deadline": csv_meta.get("bid_deadline", None),

                # PDF는 page가 들어오고, HWP는 None일 수 있습니다.
                "page_number": chunk.metadata.get("page", None),

                "section_title": csv_meta.get("section_title", None),
                "content": chunk.page_content,
                "file_name": csv_meta.get("file_name", file_name),
                "file_type": csv_meta.get(
                    "file_type",
                    os.path.splitext(file_name)[1].replace(".", "").lower()
                )
            }
        }

        rag_data.append(item)

    return rag_data


# ============================================================
# 7. 검증 함수
# ============================================================

def check_document_matching(documents, metadata_map):
    """
    로드된 Document가 CSV metadata와 매칭되는지 확인합니다.
    """

    rows = []

    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "")
        file_name = os.path.basename(source)
        file_key = normalize_filename(file_name)

        csv_meta = metadata_map.get(file_key)

        rows.append({
            "doc_index": i,
            "file_name": file_name,
            "file_key": file_key,
            "matched": csv_meta is not None,
            "page": doc.metadata.get("page", None),
        })

    doc_match_df = pd.DataFrame(rows)

    print("Document 매칭 성공 수:", doc_match_df["matched"].sum())
    print("Document 매칭 실패 수:", len(doc_match_df) - doc_match_df["matched"].sum())

    failed_df = doc_match_df[doc_match_df["matched"] == False]

    if len(failed_df) > 0:
        print("\n매칭 실패 문서:")
        print(failed_df)

    return doc_match_df


def check_null_values(rag_data):
    """
    주요 metadata 필드의 null 개수를 확인합니다.
    """

    check_fields = [
        "organization",
        "budget",
        "announcement_date",
        "bid_start",
        "bid_deadline",
        "section_title",
        "page_number",
    ]

    summary_rows = []

    for field in check_fields:
        null_count = sum(
            1 for item in rag_data
            if item["metadata"].get(field) is None
            or pd.isna(item["metadata"].get(field))
        )

        summary_rows.append({
            "field": field,
            "null_count": null_count,
            "total": len(rag_data),
            "null_ratio": null_count / len(rag_data) if rag_data else None
        })

    null_summary_df = pd.DataFrame(summary_rows)

    print("\nNull 값 체크:")
    print(null_summary_df)

    return null_summary_df


# ============================================================
# 8. 메인 실행
# ============================================================

def main():
    print("현재 작업 경로:", os.getcwd())
    print("스크립트 기준 경로:", BASE_DIR)
    print("data 폴더 경로:", DATA_DIR)
    print("CSV 파일 경로:", CSV_PATH)
    print("data 폴더 존재:", os.path.exists(DATA_DIR))
    print("CSV 파일 존재:", os.path.exists(CSV_PATH))

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"data 폴더가 없습니다: {DATA_DIR}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {CSV_PATH}")

    # 1. CSV metadata 매칭
    metadata_map, match_df = load_metadata_csv_matched(
        csv_path=CSV_PATH,
        data_dir=DATA_DIR,
        threshold=MATCH_THRESHOLD
    )

    print("\nmetadata_map 개수:", len(metadata_map))
    print("매칭 성공 수:", match_df["matched"].sum())
    print("매칭 실패 수:", len(match_df) - match_df["matched"].sum())
    print("\n매칭 결과:")
    print(match_df)

    match_df.to_csv(
        MATCH_RESULT_CSV_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("metadata_match_result.csv 저장 완료:", MATCH_RESULT_CSV_PATH)

    # 2. 문서 로드
    documents = load_documents(DATA_DIR)

    # 3. Document 기준 매칭 확인
    check_document_matching(documents, metadata_map)

    # 4. Chunk size 실험
    experiment_df = chunk_size_experiment(
        documents,
        start=100,
        end=1000,
        step=100,
        overlap_ratio=0.1
    )

    experiment_df.to_csv(
        CHUNK_EXPERIMENT_CSV_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("chunk_size_experiment_results.csv 저장 완료:", CHUNK_EXPERIMENT_CSV_PATH)

    # 5. 최종 chunk 생성
    selected_chunk_overlap = int(SELECTED_CHUNK_SIZE * 0.1)

    selected_chunks = create_chunks(
        documents,
        chunk_size=SELECTED_CHUNK_SIZE,
        chunk_overlap=selected_chunk_overlap
    )

    # 6. RAG JSON 생성
    rag_data = convert_chunks_to_rag_format(
        selected_chunks,
        metadata_map=metadata_map
    )

    print("\nselected_chunk_size:", SELECTED_CHUNK_SIZE)
    print("selected_chunk_overlap:", selected_chunk_overlap)
    print("selected_chunks 수:", len(selected_chunks))
    print("rag_data 수:", len(rag_data))

    if rag_data:
        print("\n첫 번째 RAG 데이터 미리보기:")
        print(json.dumps(rag_data[0], ensure_ascii=False, indent=2)[:3000])

    # 7. null 값 확인
    check_null_values(rag_data)

    # 8. JSON 저장
    with open(RAG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            rag_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nRAG JSON 저장 완료:", RAG_JSON_PATH)

    with open(RAG_SAMPLE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            rag_data[:5],
            f,
            ensure_ascii=False,
            indent=2
        )

    print("샘플 JSON 저장 완료:", RAG_SAMPLE_JSON_PATH)


if __name__ == "__main__":
    main()
