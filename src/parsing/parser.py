import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from src.parsing.meta_db import normalize_filename
from src.generation.gen import generate_pure_text

# 로거 설정
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_semantic_embedding() -> HuggingFaceEmbeddings:
    """
    Semantic 청킹에 사용되는 경량 임베딩 모델을 지연 로드
    """
    logger.info("Semantic chunking을 위한 로컬 임베딩 모델 로드 중...")
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")


def generate_chunk_context(
    whole_doc_text: str, 
    chunk_content: str, 
    provider: str = "local"
) -> str:
    """
    전체 문서 맥락을 바탕으로 현재 청크의 요약된 맥락(Context)을 생성
    """
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
        logger.warning(f"Contextual 맥락 생성 실패로 빈 문자열을 반환합니다. 에러: {e}")
        return ""


def inject_context_to_document(
    chunk_doc: Document, 
    whole_doc_text: str, 
    provider: str
) -> None:
    """
    생성된 맥락 요약문을 원본 청크 내용 상단에 주입
    """
    context_summary = generate_chunk_context(whole_doc_text, chunk_doc.page_content, provider=provider)
    
    if context_summary:
        logger.debug(f"Context 생성 완료: {context_summary[:30]}...")
        chunk_doc.page_content = f"[본 청크의 행정 맥락: {context_summary}]\n\n{chunk_doc.page_content}"
        chunk_doc.metadata["context_summary"] = context_summary
    else:
        chunk_doc.metadata["context_summary"] = "맥락 없음"


def create_chunks(
    documents: List[Document], 
    chunk_mode: str = "recursive", 
    chunk_size: int = 500, 
    chunk_overlap: int = 50, 
    semantic_threshold: int = 90, 
    sem_rec_chunksize: int = 1200, 
    sem_rec_overlap: int = 120, 
    sentences_per_chunk: int = 3, 
    sentence_overlap: int = 1, 
    embed_provider: str = "local", 
    use_contextual: bool = False
) -> List[Document]:
    """
    설정된 모드에 따라 문서 청킹
    """
    logger.info(f"문서 청킹 시작 (모드: {chunk_mode}, 문서 수: {len(documents)})")
    
    if chunk_mode == "recursive":
        return recursive_chunk(documents, chunk_size, chunk_overlap, use_contextual, embed_provider)
    elif chunk_mode == "semantic":
        return semantic_chunk(documents, semantic_threshold, sem_rec_chunksize, sem_rec_overlap, use_contextual, embed_provider)
    elif chunk_mode == "sentence":
        return sentence_chunk(documents, sentences_per_chunk, sentence_overlap, use_contextual, embed_provider)
    else:
        raise ValueError(f"지원하지 않는 chunk_mode: '{chunk_mode}'")


def recursive_chunk(
    documents: List[Document], 
    chunk_size: int, 
    chunk_overlap: int, 
    use_contextual: bool, 
    embed_provider: str
) -> List[Document]:
    """
    재귀적 문자 분할(Recursive) 방식으로 청킹
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    llm_provider = "openai" if embed_provider == "openai" else "local"
    final_chunks = []

    for doc in documents:
        split_chunks = splitter.split_documents([doc])
        for chunk in split_chunks:
            if use_contextual:
                inject_context_to_document(chunk, doc.page_content, provider=llm_provider)
            else:
                chunk.metadata["context_summary"] = "기능 꺼짐"
            final_chunks.append(chunk)
            
    return final_chunks


def semantic_chunk(
    documents: List[Document], 
    semantic_threshold: int, 
    sem_rec_chunksize: int, 
    sem_rec_overlap: int, 
    use_contextual: bool, 
    embed_provider: str
) -> List[Document]:
    """
    의미론적(Semantic) 분할을 우선 수행하고, 
    지나치게 긴 청크는 재귀적(Recursive)으로 다시 분할
    """
    embedding_model = get_semantic_embedding()
    splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=semantic_threshold
    )
    safe_splitter = RecursiveCharacterTextSplitter(
        chunk_size=sem_rec_chunksize,
        chunk_overlap=sem_rec_overlap
    )

    final_chunks = []
    llm_provider = "openai" if embed_provider == "openai" else "local"

    for doc in documents:
        semantic_chunks = splitter.split_documents([doc])
        
        for chunk in semantic_chunks:
            # 토큰 한도 초과 방지 (4000자 이상 시 강제 분할)
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


def sentence_chunk(
    documents: List[Document], 
    sentences_per_chunk: int, 
    sentence_overlap: int, 
    use_contextual: bool, 
    embed_provider: str
) -> List[Document]:
    """
    문장 단위로 문서를 분할하여 묶기
    """
    chunks = []
    llm_provider = "openai" if embed_provider == "openai" else "local"

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=400,
        chunk_overlap=50
    )

    for doc in documents:
        split_docs = splitter.split_documents([doc])
        step = max(1, sentences_per_chunk - sentence_overlap)
        
        for i in range(0, len(split_docs), step):
            chunk_group = split_docs[i:i + sentences_per_chunk]
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


def convert_chunks_to_rag_format(
    chunks: List[Document], 
    metadata_map: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    생성된 Chunk Document 리스트를 Vector DB 적재용 JSON(Dict) 포맷으로 변환
    """
    rag_data = []
    chunk_count_map = {}
    metadata_map = metadata_map or {}

    for chunk in chunks:
        source_path = chunk.metadata.get("source", "")
        file_name = Path(source_path).name
        file_key = normalize_filename(file_name)

        csv_meta = metadata_map.get(file_key, {})
        doc_id = str(csv_meta.get("doc_id", Path(file_name).stem))

        chunk_count_map[doc_id] = chunk_count_map.get(doc_id, 0) + 1
        chunk_id = f"{doc_id}_{chunk_count_map[doc_id]:04d}"

        # 파일 확장자 추출 (ex: .pdf -> pdf)
        file_ext = Path(file_name).suffix.replace(".", "").lower()

        item = {
            "id": chunk_id,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": csv_meta.get("title", Path(file_name).stem),
                "organization": csv_meta.get("organization"),
                "budget": csv_meta.get("budget"),
                "announcement_date": csv_meta.get("announcement_date"),
                "bid_start": csv_meta.get("bid_start"),
                "bid_deadline": csv_meta.get("bid_deadline"),
                "page_number": chunk.metadata.get("page"),
                "section_title": csv_meta.get("section_title"),
                "content": chunk.page_content,
                "file_name": csv_meta.get("file_name", file_name),
                "file_type": csv_meta.get("file_type", file_ext),
                "context_summary": chunk.metadata.get("context_summary"),
            },
        }
        rag_data.append(item)

    return rag_data


def check_document_matching(documents: List[Document], metadata_map: Dict[str, Any]) -> pd.DataFrame:
    """
    문서와 메타데이터의 매칭 성공 여부를 점검하고 통계 반환
    """
    rows = []
    for i, doc in enumerate(documents):
        file_name = Path(doc.metadata.get("source", "")).name
        file_key = normalize_filename(file_name)

        rows.append({
            "doc_index": i,
            "file_name": file_name,
            "file_key": file_key,
            "matched": file_key in metadata_map,
            "page": doc.metadata.get("page"),
        })

    doc_match_df = pd.DataFrame(rows)
    success_count = doc_match_df["matched"].sum()
    fail_count = len(doc_match_df) - success_count

    logger.info(f"문서 청크 매칭 결과: 성공 {success_count}건, 실패 {fail_count}건")
    
    if fail_count > 0:
        failed_df = doc_match_df[~doc_match_df["matched"]]
        logger.warning(f"매칭 실패 문서 일부:\n{failed_df.head()}")

    return doc_match_df


def check_null_values(rag_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    최종 RAG 데이터 내 필수 메타데이터 컬럼의 Null 비율 점검
    """
    check_fields = [
        "organization", "budget", "announcement_date",
        "bid_start", "bid_deadline", "section_title", "page_number",
    ]
    summary_rows = []
    total_len = len(rag_data)

    for field in check_fields:
        null_count = sum(
            1 for item in rag_data
            if pd.isna(item["metadata"].get(field)) or item["metadata"].get(field) is None
        )
        summary_rows.append({
            "field": field,
            "null_count": null_count,
            "total": total_len,
            "null_ratio": (null_count / total_len) if total_len > 0 else 0,
        })

    null_summary_df = pd.DataFrame(summary_rows)
    logger.info(f"Null 값 체크 완료:\n{null_summary_df}")

    return null_summary_df