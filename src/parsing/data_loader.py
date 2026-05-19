"""
PDF/HWP 문서를 LangChain Document 형태로 로드합니다.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from src.parsing.hwp_loader import HWPLoader
from src.parsing.meta_db import normalize_source_filename


def _stamp_source_metadata(documents, file_name, file_type):
    source_file_name = normalize_source_filename(file_name)
    for doc in documents:
        doc.metadata["source"] = source_file_name
        doc.metadata["file_name"] = source_file_name
        doc.metadata["file_type"] = file_type
    return documents


def load_documents(data_dir: str):
    documents = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data 폴더가 없습니다: {data_dir}")

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)

        if not os.path.isfile(file_path):
            continue

        if file.lower().endswith(".pdf"):
            print("PDF 로드:", file)
            try:
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(_stamp_source_metadata(loaded_docs, file, "pdf"))
            except Exception as e:
                print("PDF 로드 실패:", file)
                print(e)

        elif file.lower().endswith(".hwp"):
            print("HWP 로드:", file)
            try:
                loader = HWPLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(_stamp_source_metadata(loaded_docs, file, "hwp"))
            except Exception as e:
                print("HWP 로드 실패:", file)
                print(e)

        # else:
        #     print("제외:", file)

    print("\n전체 로드된 Document 수:", len(documents))
    return documents
