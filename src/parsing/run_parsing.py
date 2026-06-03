import logging
from pathlib import Path
from typing import Any, Dict, List

from src.parsing.data_loader import load_documents
from src.parsing.meta_db import load_metadata_db
from src.parsing.parser import create_chunks, convert_chunks_to_rag_format
from src.parsing.ocr import OCR_parsing

# 로거 설정
logger = logging.getLogger(__name__)


def run_parsing(
    run_ocr: bool,
    chunk_mode: str = "semantic",
    use_contextual: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    semantic_threshold: int = 80,
    sem_rec_chunksize: int = 1200,
    sem_rec_overlap: int = 100,
    sentences_per_chunk: int = 3,
    sentence_overlap: int = 1,
    match_threshold: float = 0.55,
    embed_provider: str = "local",
    data_dir: str = "data",
    csv_path: str = "data/data_list.csv"
) -> List[Dict[str, Any]]:
    """
    문서 파싱 파이프라인을 실행하고 Vector DB 적재용 RAG 데이터 반환
    """
    data_path = Path(data_dir)
    csv_file = Path(csv_path)

    # 1. 대상 폴더 및 메타데이터 파일 존재 여부 검증
    if not data_path.is_dir():
        logger.error(f"데이터 폴더가 없습니다. 경로: {data_path.resolve()}")
        raise FileNotFoundError(f"data 폴더를 찾을 수 없습니다: {data_path}")

    if not csv_file.is_file():
        logger.error(f"메타데이터 CSV 파일이 없습니다. 경로: {csv_file.resolve()}")
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file}")

    logger.info("--- 파싱 파이프라인 시작 ---")

    # 2. 메타데이터 DB 로드 및 파일명 매칭
    try:
        metadata_map, _ = load_metadata_db(
            csv_path=csv_file,
            data_dir=data_path,
            threshold=match_threshold,
        )
    except Exception as e:
        logger.error(f"메타데이터 매칭 단계 중 오류 발생: {e}", exc_info=True)
        raise

    # 3. 문서 로드 (PDF, HWP 등)
    try:
        documents = load_documents(data_dir=data_path)
    except Exception as e:
        logger.error(f"문서 로드 단계 중 오류 발생: {e}", exc_info=True)
        raise

    if not documents:
        logger.warning("로드된 문서가 없습니다. data 폴더 내 파일들을 확인하세요.")
        return []

    # 4. 문서 청킹 (Chunking)
    parser_kwargs = {
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "semantic_threshold": semantic_threshold,
        "sem_rec_chunksize": sem_rec_chunksize,
        "sem_rec_overlap": sem_rec_overlap,
        "sentences_per_chunk": sentences_per_chunk,
        "sentence_overlap": sentence_overlap,
        "embed_provider": embed_provider,
        "use_contextual": use_contextual
    }
    try:
        if run_ocr:
            rag_data = OCR_parsing(
                documents=documents,
                data_dir=data_dir,
                csv_path=csv_path,
                match_threshold=match_threshold,
                **parser_kwargs  # 딕셔너리로 깔끔하게 전달!
            )
        else:
            try:
                chunks = create_chunks(documents=documents, **parser_kwargs)

            except Exception as e:
                logger.error(f"RAG 포맷 변환 단계 중 오류 발생: {e}", exc_info=True)
                raise

            rag_data = convert_chunks_to_rag_format(chunks, metadata_map=metadata_map)

    except Exception as e:
        logger.error(f"문서 청킹 단계 중 오류 발생: {e}", exc_info=True)
        raise

    logger.info(f"파싱 파이프라인 완료: 총 {len(rag_data)}개 청크 데이터가 생성되었습니다.")
    return rag_data