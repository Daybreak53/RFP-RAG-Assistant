import logging
import olefile
import zlib
import struct
import re
import unicodedata
from typing import Any, List, Iterator, Tuple

from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader

# 로거 설정
logger = logging.getLogger(__name__)

class HWPLoader(BaseLoader):
    """
    HWP 파일 읽기 클래스
    """

    FILE_HEADER_SECTION = "FileHeader"
    HWP_SUMMARY_SECTION = "\x05HwpSummaryInformation"
    BODYTEXT_SECTION = "BodyText"
    SECTION_NAME_PREFIX_LENGTH = len("Section")
    HWP_TEXT_TAGS = frozenset([67])

    def __init__(self, file_path: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.extra_info = {"source": file_path}

    def lazy_load(self) -> Iterator[Document]:
        """
        HWP 파일에서 텍스트를 로드하여 LangChain Document로 반환
        """
        logger.debug(f"HWP 문서 로드 시도: {self.file_path}")
        
        try:
            # 컨텍스트 매니저를 사용하여 안전한 파일 핸들링
            with olefile.OleFileIO(self.file_path) as load_file:
                file_dir = load_file.listdir()

                if not self._is_valid_hwp(file_dir):
                    raise ValueError(f"유효하지 않은 HWP OLE 구조입니다: {self.file_path}")

                result_text = self._extract_text(load_file, file_dir)
                yield Document(page_content=result_text, metadata=self.extra_info.copy())
                
        except Exception as e:
            logger.error(f"HWP 파일 파싱 중 오류 발생 ({self.file_path}): {e}", exc_info=True)
            raise

    def _is_valid_hwp(self, dirs: List[List[str]]) -> bool:
        """
        HWP 파일의 필수 OLE 디렉토리 존재 여부 검사
        """
        return [self.FILE_HEADER_SECTION] in dirs and [self.HWP_SUMMARY_SECTION] in dirs

    def _get_body_sections(self, dirs: List[List[str]]) -> List[str]:
        """
        문서 내의 본문(BodyText) 섹션 경로 목록을 숫자에 맞춰 정렬하여 반환
        """
        section_numbers = []
        for d in dirs:
            if d[0] == self.BODYTEXT_SECTION and len(d) > 1:
                try:
                    num = int(d[1][self.SECTION_NAME_PREFIX_LENGTH:])
                    section_numbers.append(num)
                except ValueError:
                    continue
                
        return [f"{self.BODYTEXT_SECTION}/Section{num}" for num in sorted(section_numbers)]

    def _extract_text(self, load_file: olefile.OleFileIO, file_dir: List[List[str]]) -> str:
        """
        모든 섹션에서 텍스트를 추출하여 하나의 문자열로 결합
        """
        is_compressed = self._is_compressed(load_file)
        sections = self._get_body_sections(file_dir)
        
        texts = []
        for section in sections:
            try:
                text = self._get_text_from_section(load_file, section, is_compressed)
                if text:
                    texts.append(text)
            except Exception as e:
                logger.warning(f"HWP 섹션 파싱 실패, 해당 섹션 건너뜀 ({section}): {e}")
                
        return "\n".join(texts)

    def _is_compressed(self, load_file: olefile.OleFileIO) -> bool:
        """
        파일 헤더를 읽어 압축(Deflate) 적용 여부 확인
        """
        with load_file.openstream(self.FILE_HEADER_SECTION) as header:
            header_data = header.read()
            return bool(header_data[36] & 1)

    def _get_text_from_section(self, load_file: olefile.OleFileIO, section: str, is_compressed: bool) -> str:
        """
        특정 스트림(섹션)의 바이너리를 읽어 텍스트로 디코딩
        """
        with load_file.openstream(section) as bodytext:
            data = bodytext.read()

        if is_compressed:
            try:
                # -15는 zlib에서 raw deflate 포맷을 읽기 위한 wbits 설정값
                unpacked_data = zlib.decompress(data, -15)
            except zlib.error as e:
                logger.error(f"HWP 섹션 압축 해제 실패 ({section}): {e}")
                return ""
        else:
            unpacked_data = data

        text_chunks = []
        i = 0
        data_length = len(unpacked_data)
        
        while i < data_length:
            if i + 4 > data_length:
                break  # 헤더(4 bytes)를 읽을 공간 부족
                
            header, rec_type, rec_len = self._parse_record_header(unpacked_data[i : i + 4])
            
            if rec_type in self.HWP_TEXT_TAGS:
                if i + 4 + rec_len > data_length:
                    break  # 레코드 데이터를 읽을 공간 부족
                    
                rec_data = unpacked_data[i + 4 : i + 4 + rec_len]
                try:
                    text_chunks.append(rec_data.decode("utf-16"))
                except UnicodeDecodeError:
                    pass  # 디코딩 실패 시 해당 청크 무시
                    
            i += 4 + rec_len

        # 병합 후 필터링 진행
        raw_text = "\n".join(text_chunks)
        clean_text = self.remove_chinese_characters(raw_text)
        clean_text = self.remove_control_characters(clean_text)
        
        return clean_text

    @staticmethod
    def remove_chinese_characters(s: str) -> str:
        """
        한자(중국어) 문자를 정규식으로 제거
        """
        return re.sub(r"[\u4e00-\u9fff]+", "", s)

    @staticmethod
    def remove_control_characters(s: str) -> str:
        """
        출력 시 깨지거나 불필요한 제어 문자 제거
        """
        # 카테고리 'C' (Control, Format, Unassigned, Private Use, Surrogate) 제외
        return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

    @staticmethod
    def _parse_record_header(header_bytes: bytes) -> Tuple[int, int, int]:
        """4바이트 레코드 헤더를 파싱하여 (전체값, 타입, 길이)를 반환합니다."""
        header = struct.unpack_from("<I", header_bytes)[0]
        rec_type = header & 0x3FF           # 하위 10비트
        rec_len = (header >> 20) & 0xFFF    # 상위 12비트
        return header, rec_type, rec_len