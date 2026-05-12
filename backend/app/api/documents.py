import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.schemas.documents import DocumentResponse, DeleteResponse
from app.services import document_service

router = APIRouter()

@router.get("", response_model=List[DocumentResponse])
async def list_documents():
    """업로드된 문서 목록 조회"""
    return document_service.get_all_documents()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """문서 업로드 처리"""
    await asyncio.sleep(0.5)

    file_size = file.size if file.size is not None else 0
    return document_service.save_document(file.filename, file_size)

@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """문서 삭제"""
    success = document_service.remove_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found") 
    
    return DeleteResponse(status="success", id=doc_id)