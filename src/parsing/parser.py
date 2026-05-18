"""
문서 청킹, chunk size 실험, RAG JSON 형식 변환 모듈입니다.
"""

import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.parsing.meta_db import normalize_filename
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import re

embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small") #semantic chunking 용 임베딩 모델(경량화되어있음)

def create_chunks(documents, chunk_mode="recursive", chunk_size=500, chunk_overlap=50, semantic_threshold=90, sentences_per_chunk=3, sentence_overlap=1):
    if chunk_mode == "recursive":
        return recursive_chunk(documents, chunk_size, chunk_overlap)
    elif chunk_mode == "semantic":
        return semantic_chunk(documents, semantic_threshold)
    elif chunk_mode == "sentence":
        return sentence_chunk(documents, sentences_per_chunk, sentence_overlap)
    else:
        raise ValueError(f"지원하지 않는 chunk_mode: '{chunk_mode}'")

def recursive_chunk(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)

def semantic_chunk(documents, semantic_threshold):

    splitter = SemanticChunker(
        embedding,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=semantic_threshold
    )
    return splitter.split_documents(documents)


def sentence_chunk(documents, sentences_per_chunk, sentence_overlap):
    chunks = []
    def simple_sentence_split(text):
        sentences = re.split(r'(?<=[.!?。])\s+', text)

        return [s.strip() for s in sentences if s.strip()]

    for doc in documents:
        text = doc.page_content

        sentences = simple_sentence_split(text)

        step = max(1, sentences_per_chunk - sentence_overlap) # 오버랩과 문장수가 동일해질 때 무한루프 방지
        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i+sentences_per_chunk]
            if not chunk_sentences:
                continue
            chunk_text = " ".join(chunk_sentences)

            if len(chunk_text) > 4000:
                chunk_text = chunk_text[:4000]

            chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))

    return chunks


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

#밑에 2개의 실험은 어디서 함수 적용하는지 모르겠어서 일단 만들어놨습니다.

#semantic_chuink 기법 threshold 별 실험
"""
def semantic_chunk_experiment(
    documents,
    thresholds=[70, 80, 90, 95]
):
    results = []

    for threshold in thresholds:

        chunks = create_chunks(
            documents,
            chunk_mode="semantic",
            semantic_threshold=threshold
        )

        chunk_lengths = [
            len(chunk.page_content)
            for chunk in chunks
        ]

        results.append({
            "semantic_threshold": threshold,
            "num_chunks": len(chunks),
            "avg_chunk_length": (
                sum(chunk_lengths) / len(chunk_lengths)
                if chunk_lengths else 0
            ),
            "max_chunk_length": (
                max(chunk_lengths)
                if chunk_lengths else 0
            ),
            "min_chunk_length": (
                min(chunk_lengths)
                if chunk_lengths else 0
            ),
        })

        print("=" * 50)
        print("semantic_threshold:", threshold)
        print("num_chunks:", len(chunks))
        print("avg_chunk_length:", results[-1]["avg_chunk_length"])

    return pd.DataFrame(results)
"""

#sentence_chunk 기법 사이즈, 오버랩 별 실험
"""
def sentence_chunk_experiment(
    documents,
    sentence_sizes=[3, 5, 7],
    overlaps=[1, 2]
):
    results = []

    for sentence_size in sentence_sizes:

        for overlap in overlaps:

            if overlap >= sentence_size:
                continue

            chunks = create_chunks(
                documents,
                chunk_mode="sentence",
                sentences_per_chunk=sentence_size,
                sentence_overlap=overlap
            )

            chunk_lengths = [
                len(chunk.page_content)
                for chunk in chunks
            ]

            results.append({
                "sentences_per_chunk": sentence_size,
                "sentence_overlap": overlap,
                "num_chunks": len(chunks),
                "avg_chunk_length": (
                    sum(chunk_lengths) / len(chunk_lengths)
                    if chunk_lengths else 0
                ),
                "max_chunk_length": (
                    max(chunk_lengths)
                    if chunk_lengths else 0
                ),
                "min_chunk_length": (
                    min(chunk_lengths)
                    if chunk_lengths else 0
                ),
            })

            print("=" * 50)
            print("sentences_per_chunk:", sentence_size)
            print("sentence_overlap:", overlap)
            print("num_chunks:", len(chunks))
            print("avg_chunk_length:", results[-1]["avg_chunk_length"])

    return pd.DataFrame(results)
"""

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
