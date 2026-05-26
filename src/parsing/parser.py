"""
문서 청킹, chunk size 실험, RAG JSON 형식 변환 모듈입니다.
"""

import os
import re
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from src.parsing.meta_db import normalize_filename
from src.generation.gen import generate_pure_text


embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small") #semantic chunking 용 임베딩 모델(경량화되어있음)

def generate_chunk_context(whole_doc_text: str, chunk_content: str, provider: str = "local") -> str:
    prompt = f"""너는 대한민국 공공기관 및 재단법인의 '입찰공고문'과 '제안요청서(RFP)' 분석 전문가이다.
주어진 [전체 문서]의 거시적 맥락을 바탕으로 해당 청크가 문서 내에서 어떤 역할이나 규정을 담고 있는지 1~2문장의 정제된 행정학적 문장으로 요약해라.

[작성 규칙]
1. 이 청크가 무엇에 관한 규정인지 명확하게 명시할 것 (예: 본 청크는 사업비 정산 및 지출 증빙에 관한 규정임).
2. 수식어나 감정적 표현은 철저히 배제하고, 실제 공공기관 보고서 서식에 들어갈 법한 딱딱하고 건조한 문체를 사용할 것.
3. 답변은 수식어 없이 오직 요약된 1~2문단(문장)만 담백하게 출력할 것.

[전체 문서]
{whole_doc_text}

[행정 청크]
{chunk_content}

가상 맥락 요약 문장:"""

    try:
        context_summary = generate_pure_text(prompt, provider=provider)
        return context_summary.strip()
    except Exception as e:
        print(f"[경고] Contextual 맥락 생성 실패로 빈 문자열을 반환합니다. 에러: {e}")
        return ""

def inject_context_to_document(chunk_doc, whole_doc_text: str, provider: str) -> None:
    # 1. 맥락 요약문 생성
    context_summary = generate_chunk_context(whole_doc_text, chunk_doc.page_content, provider=provider)
    
    # 2. content 가공 및 복사)
    if context_summary:
        print(f"[HyDE-Context 생성 문장]: {context_summary}")
        chunk_doc.page_content = f"[본 청크의 행정 맥락: {context_summary}]\n\n{chunk_doc.page_content}"
        chunk_doc.metadata["context_summary"] = context_summary
    else:
        chunk_doc.metadata["context_summary"] = "맥락 없음"


def create_chunks(documents, chunk_mode="recursive", chunk_size=500, chunk_overlap=50, semantic_threshold=90, sem_rec_chunksize=1200, sem_rec_overlap=120, sentences_per_chunk=3, sentence_overlap=1, embed_provider="local", use_contextual=False):
    if chunk_mode == "recursive":
        return recursive_chunk(documents, chunk_size, chunk_overlap, use_contextual, embed_provider=embed_provider)
    elif chunk_mode == "semantic":
        return semantic_chunk(documents, semantic_threshold, sem_rec_chunksize, sem_rec_overlap, use_contextual, embed_provider=embed_provider)
    elif chunk_mode == "sentence":
        return sentence_chunk(documents, sentences_per_chunk, sentence_overlap, use_contextual, embed_provider=embed_provider)
    else:
        raise ValueError(f"지원하지 않는 chunk_mode: '{chunk_mode}'")

def recursive_chunk(documents, chunk_size, chunk_overlap, use_contextual, embed_provider):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    llm_provider = "openai" if embed_provider == "openai" else "local"
    final_chunks = []

    for doc in documents:
        split_chunks = splitter.split_documents([doc])
        
        for chunk in split_chunks:
            # use_contextual 플래그가 True일 때만 LLM을 호출하여 맥락 주입
            if use_contextual:
                inject_context_to_document(chunk, doc.page_content, provider=llm_provider)
            else:
                # 기능을 껐을 때도 구조를 맞추기 위해 메타데이터만 기본값 세팅
                chunk.metadata["context_summary"] = "기능 꺼짐"
                
            final_chunks.append(chunk)
            
    return final_chunks

#의미기반으로만 나누면 자꾸 OPENAI의 8192개 토큰을 넘어가버려 semantic + recursive 구조로 변경
def semantic_chunk(documents, semantic_threshold, sem_rec_chunksize, sem_rec_overlap, use_contextual, embed_provider):

    splitter = SemanticChunker(
        _get_embedding(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=semantic_threshold
    )
    semantic_chunks = splitter.split_documents(documents)

    safe_splitter = RecursiveCharacterTextSplitter(
        chunk_size=sem_rec_chunksize,
        chunk_overlap=sem_rec_overlap
    )

    final_chunks = []
    llm_provider = "openai" if embed_provider == "openai" else "local"

    for doc in documents:
        semantic_chunks = splitter.split_documents([doc])
        
        for chunk in semantic_chunks:
            if len(chunk.page_content) > 4000:
                split_chunks = safe_splitter.split_documents([chunk])
                for s_chunk in split_chunks:
                    if use_contextual:
                        inject_context_to_document(s_chunk, doc.page_content, provider=llm_provider)
                    else:
                        s_chunk.metadata["context_summary"] = "기능 꺼짐"
                    final_chunks.append(s_chunk)
            else:
                if use_contextual:
                    inject_context_to_document(chunk, doc.page_content, provider=llm_provider)
                else:
                    chunk.metadata["context_summary"] = "기능 꺼짐"
                final_chunks.append(chunk)

    return final_chunks


def sentence_chunk(documents, sentences_per_chunk, sentence_overlap, use_contextual, embed_provider):
    chunks = []
    llm_provider = "openai" if embed_provider == "openai" else "local"

    for doc in documents:
        #sentence 청킹 수정사항.
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=400,
            chunk_overlap=50
        )
        
        split_docs = splitter.split_documents([doc])
        
        step = max(1, sentences_per_chunk - sentence_overlap)
        
        for i in range(0, len(split_docs), step):
            chunk_group = split_docs[i:i+sentences_per_chunk]
            if not chunk_group:
                continue
                
            chunk_text = " ".join([d.page_content.strip() for d in chunk_group])
            
            if len(chunk_text) > 4000:
                chunk_text = chunk_text[:4000]
                
            chunk = Document(page_content=chunk_text, metadata=doc.metadata.copy())
            
            if use_contextual:
                inject_context_to_document(chunk, doc.page_content, provider=llm_provider)
            else:
                chunk.metadata["context_summary"] = "기능 꺼짐"
                
            chunks.append(chunk)

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
        file_name = _resolve_chunk_source_filename(chunk.metadata)
        if not file_name:
            raise ValueError("청크 메타데이터에서 원본 HWP/PDF 파일명을 찾을 수 없습니다.")

        file_key = normalize_filename(file_name)

        csv_meta = metadata_map.get(file_key, {}) if metadata_map else {}
        doc_id = csv_meta.get("doc_id", os.path.splitext(file_name)[0])
        file_type = os.path.splitext(file_name)[1].replace(".", "").lower()

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
                "file_name": file_name,
                "file_type": file_type,
            },
        }

        rag_data.append(item)

    return rag_data


def _resolve_chunk_source_filename(metadata):
    for key in ("file_name", "filename", "source", "file_path", "path"):
        file_name = normalize_source_filename(metadata.get(key, ""))
        if os.path.splitext(file_name)[1].lower() in {".hwp", ".pdf"}:
            return file_name
    return ""


def check_document_matching(documents, metadata_map):
    rows = []

    for i, doc in enumerate(documents):
        file_name = _resolve_chunk_source_filename(doc.metadata)
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
