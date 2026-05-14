"""
PDF/HWP 문서를 LangChain Document 형태로 로드합니다.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from src.parsing.hwp_loader import HWPLoader


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
