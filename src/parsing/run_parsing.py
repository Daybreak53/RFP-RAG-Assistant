import os
import json
from src.parsing.data_loader import load_documents
from src.parsing.meta_db import load_metadata_db
from src.parsing.parser import (
    create_chunks,
    convert_chunks_to_rag_format,
)

def run_parsing(chunk_mode=None, chunk_size=None, chunk_overlap=None, semantic_threshold=None, sentences_per_chunk=None, sentence_overlap=None, match_threshold=None, ):
    data_dir = "data"
    csv_path = "data/data_list.csv"

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data 폴더가 없습니다: {data_dir}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    metadata_map, _ = load_metadata_db(
        csv_path=csv_path,
        data_dir=data_dir,
        threshold=match_threshold,
    )

    documents = load_documents(data_dir)

    chunks = create_chunks(
        documents,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        
        semantic_threshold=semantic_threshold,

        sentences_per_chunk=sentences_per_chunk,
        sentence_overlap=sentence_overlap,
    )

    rag_data = convert_chunks_to_rag_format(chunks, metadata_map=metadata_map)

    print(f"파싱 완료: {len(rag_data)}개 청크 생성됨")
    return rag_data