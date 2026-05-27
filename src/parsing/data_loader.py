import logging
import unicodedata

from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from src.parsing.hwp_loader import HWPLoader

# 로거 설정
logger = logging.getLogger(__name__)


def load_documents(data_dir: Union[str, Path]) -> List[Document]:
    """
    지정된 디렉토리 내의 지원되는 파일들을 읽어 LangChain Document 리스트로 반환
    """
    data_path = Path(data_dir)
    
    if not data_path.is_dir():
        logger.error(f"데이터 디렉토리를 찾을 수 없습니다: {data_path.resolve()}")
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {data_path}")

    documents: List[Document] = []

    # 확장자별 Loader 매핑 
    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".hwp": HWPLoader,
    }

    logger.info(f"문서 로드 시작 (대상 디렉토리: {data_path})")

    for file_path in data_path.iterdir():
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()

        # 지원하는 확장자인지 검사
        if ext in loader_mapping:
            safe_name = unicodedata.normalize('NFC', file_path.name)
            logger.info(f"{ext.upper()[1:]} 로드 중: {safe_name}")
            loader_class = loader_mapping[ext]
            
            try:
                # 로더 초기화 및 문서 로드
                loader = loader_class(str(file_path))
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                
                logger.debug(f"성공적으로 로드됨: {file_path.name} ({len(loaded_docs)} 페이지/청크)")
            except Exception as e:
                logger.error(f"{ext.upper()[1:]} 문서 로드 실패 ({file_path.name}): {e}", exc_info=True)

    logger.info(f"문서 로드 완료 (전체 로드된 Document 수: {len(documents)})")
    
    return documents