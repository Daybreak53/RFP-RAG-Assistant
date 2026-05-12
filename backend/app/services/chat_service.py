import asyncio
import json
from typing import Optional
from app.schemas.chat import ChatOptions # 추가 임포트

async def generate_stream(message: str, options: Optional[ChatOptions] = None):
    """
    SSE 형식으로 청크를 반환하는 제너레이터 함수
    """
    mock_answer = f"안녕하세요! 전달해주신 질문 **{message}**에 대한 백엔드(FastAPI) 스트리밍 응답입니다.\n\n이 템플릿을 바탕으로 실제 RAG 로직을 구현하시면 됩니다."
    
    opts = options or ChatOptions()
    
    chunk_size = 2
    for i in range(0, len(mock_answer), chunk_size):
        chunk = mock_answer[i:i+chunk_size]
        data = {"type": "content", "text": chunk}
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.04)
        
    metadata = {
        "type": "metadata",
        "model": opts.model,
        "temperature": opts.temperature,
        "maxTokens": opts.maxTokens,
        "searchMethod": opts.searchMethod,
        "topK": opts.topK,
        "reranker": opts.reranker,
        "rerankTopK": opts.rerankTopK,
        "cached": opts.semanticCache,
        "splitMethod": opts.splitMethod,
        "chunkSize": opts.chunkSize,
        "chunkOverlap": opts.chunkOverlap,
        "streamResponse": opts.streamResponse,
        "tokensUsed": 256,
        "responseTime": 1500,
        "sources": [
            {"document": "아키텍처_설계도.pdf", "content": "FastAPI 구조에 대한 설명입니다.", "score": 0.98},
            {"document": "사내_규정집.docx", "content": "관련 규정의 일부 내용입니다.", "score": 0.85}
        ]
    }
    yield f"data: {json.dumps(metadata, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"