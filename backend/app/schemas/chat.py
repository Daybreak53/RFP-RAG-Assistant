from pydantic import BaseModel
from typing import Optional, List

class ChatOptions(BaseModel):
    searchMethod: Optional[str] = "hybrid"
    topK: Optional[int] = 5
    reranker: Optional[bool] = False
    rerankTopK: Optional[int] = 3
    semanticCache: Optional[bool] = False
    splitMethod: Optional[str] = "recursive"
    chunkSize: Optional[int] = 512
    chunkOverlap: Optional[int] = 50
    model: Optional[str] = "gemini-3-1-flash-lite"
    temperature: Optional[float] = 0.5
    maxTokens: Optional[int] = 1024
    streamResponse: Optional[bool] = True

class ChatRequest(BaseModel):
    message: str
    options: Optional[ChatOptions] = None

class ChatSource(BaseModel):
    document: str
    content: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    model: str
    sources: List[ChatSource] = []