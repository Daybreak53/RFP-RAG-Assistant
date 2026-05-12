import uuid
from typing import List
from app.schemas.documents import DocumentResponse

mock_db: List[DocumentResponse] = []

def get_all_documents() -> List[DocumentResponse]:
    return mock_db

def save_document(name: str, size: int) -> DocumentResponse:
    new_doc = DocumentResponse(
        id=str(uuid.uuid4()),
        name=name,
        size=size,
        status="ready"
    )
    mock_db.append(new_doc)
    return new_doc

def remove_document(doc_id: str) -> bool:
    global mock_db
    initial_length = len(mock_db)
    mock_db = [doc for doc in mock_db if doc.id != doc_id]
    return len(mock_db) < initial_length