"""
문서 청킹, chunk size 실험, RAG JSON 형식 변환 모듈입니다.
"""

import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.parsing.meta_db import normalize_filename


from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(documents, chunk_mode="recursive", chunk_size=500, chunk_overlap=50):
    if chunk_mode == "recursive":
        return recursive_chunk(documents, chunk_size, chunk_overlap)
    elif chunk_mode == "semantic":
        return semantic_chunk(documents)
    elif chunk_mode == "sentence":
        return sentence_chunk(documents)
    else:
        raise ValueError(f"지원하지 않는 chunk_mode: '{chunk_mode}'")

def recursive_chunk(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)

def semantic_chunk(documents):
    # TODO
    raise NotImplementedError("시맨틱 청킹 구현 필요")

def sentence_chunk(documents):
    # TODO
    raise NotImplementedError("문장 단위 청킹 구현 필요")


def chunk_size_experiment(documents, start=100, end=1000, step=100, overlap_ratio=0.1):
    results = []

    for chunk_size in range(start, end + 1, step):
        chunk_overlap = int(chunk_size * overlap_ratio)
        chunks = create_chunks(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        results.append({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": len(chunks),
        })

        print("=" * 50)
        print("chunk_size:", chunk_size)
        print("chunk_overlap:", chunk_overlap)
        print("num_chunks:", len(chunks))

    return pd.DataFrame(results)


def convert_chunks_to_rag_format(chunks, metadata_map=None):
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
        chunk_id = f"{doc_id}_{chunk_count_map[doc_id]:04d}"

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
                "page_number": chunk.metadata.get("page", None),
                "section_title": csv_meta.get("section_title", None),
                "content": chunk.page_content,
                "file_name": csv_meta.get("file_name", file_name),
                "file_type": csv_meta.get(
                    "file_type",
                    os.path.splitext(file_name)[1].replace(".", "").lower(),
                ),
            },
        }

        rag_data.append(item)

    return rag_data


def check_document_matching(documents, metadata_map):
    rows = []

    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "")
        file_name = os.path.basename(source)
        file_key = normalize_filename(file_name)

        rows.append({
            "doc_index": i,
            "file_name": file_name,
            "file_key": file_key,
            "matched": metadata_map.get(file_key) is not None,
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
            "null_ratio": null_count / len(rag_data) if rag_data else None,
        })

    null_summary_df = pd.DataFrame(summary_rows)

    print("\nNull 값 체크:")
    print(null_summary_df)

    return null_summary_df
