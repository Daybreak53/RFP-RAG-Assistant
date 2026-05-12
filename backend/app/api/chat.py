from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import generate_stream

router = APIRouter()

@router.post("", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """일반(동기) 채팅 엔드포인트"""
    return ChatResponse(
        answer=f"'{request.message}'에 대한 백엔드 일반 Mock 응답입니다.",
        model=request.options.model if request.options else "unknown",
        sources=[]
    )

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """스트리밍 채팅 엔드포인트"""
    return StreamingResponse(
        generate_stream(request.message, request.options),
        media_type="text/event-stream"
    )