"""
데이터 로드 / 메타데이터 DB / 파싱을 연결하는 실행 파일입니다.

실행:
python parsing/run_parsing.py

또는 프로젝트 루트에서:
python -m parsing.run_parsing
"""

import os
import json

from src.parsing.data_loader import load_documents
from src.parsing.meta_db import load_metadata_db
from src.parsing.parser import (
    create_chunks,
    chunk_size_experiment,
    convert_chunks_to_rag_format,
    check_document_matching,
    check_null_values,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(BASE_DIR, "data_list.csv")

RAG_JSON_PATH = os.path.join(BASE_DIR, "rag_data.json")
RAG_SAMPLE_JSON_PATH = os.path.join(BASE_DIR, "rag_data_sample.json")
MATCH_RESULT_CSV_PATH = os.path.join(BASE_DIR, "metadata_match_result.csv")
CHUNK_EXPERIMENT_CSV_PATH = os.path.join(BASE_DIR, "chunk_size_experiment_results.csv")

SELECTED_CHUNK_SIZE = 500
MATCH_THRESHOLD = 0.55


def main():
    print("현재 작업 경로:", os.getcwd())
    print("스크립트 기준 경로:", BASE_DIR)
    print("data 폴더 경로:", DATA_DIR)
    print("CSV 파일 경로:", CSV_PATH)

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"data 폴더가 없습니다: {DATA_DIR}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {CSV_PATH}")

    metadata_map, match_df = load_metadata_db(
        csv_path=CSV_PATH,
        data_dir=DATA_DIR,
        threshold=MATCH_THRESHOLD,
    )

    print("\nmetadata_map 개수:", len(metadata_map))
    print("매칭 성공 수:", match_df["matched"].sum())
    print("매칭 실패 수:", len(match_df) - match_df["matched"].sum())
    print("\n매칭 결과:")
    print(match_df)

    match_df.to_csv(MATCH_RESULT_CSV_PATH, index=False, encoding="utf-8-sig")
    print("metadata_match_result.csv 저장 완료:", MATCH_RESULT_CSV_PATH)

    documents = load_documents(DATA_DIR)
    check_document_matching(documents, metadata_map)

    experiment_df = chunk_size_experiment(
        documents,
        start=100,
        end=1000,
        step=100,
        overlap_ratio=0.1,
    )

    experiment_df.to_csv(CHUNK_EXPERIMENT_CSV_PATH, index=False, encoding="utf-8-sig")
    print("chunk_size_experiment_results.csv 저장 완료:", CHUNK_EXPERIMENT_CSV_PATH)

    selected_chunk_overlap = int(SELECTED_CHUNK_SIZE * 0.1)

    selected_chunks = create_chunks(
        documents,
        chunk_size=SELECTED_CHUNK_SIZE,
        chunk_overlap=selected_chunk_overlap,
    )

    rag_data = convert_chunks_to_rag_format(
        selected_chunks,
        metadata_map=metadata_map,
    )

    print("\nselected_chunk_size:", SELECTED_CHUNK_SIZE)
    print("selected_chunk_overlap:", selected_chunk_overlap)
    print("selected_chunks 수:", len(selected_chunks))
    print("rag_data 수:", len(rag_data))

    if rag_data:
        print("\n첫 번째 RAG 데이터 미리보기:")
        print(json.dumps(rag_data[0], ensure_ascii=False, indent=2)[:3000])

    check_null_values(rag_data)

    with open(RAG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rag_data, f, ensure_ascii=False, indent=2)

    print("\nRAG JSON 저장 완료:", RAG_JSON_PATH)

    with open(RAG_SAMPLE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rag_data[:5], f, ensure_ascii=False, indent=2)

    print("샘플 JSON 저장 완료:", RAG_SAMPLE_JSON_PATH)


if __name__ == "__main__":
    main()
