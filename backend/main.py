from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, documents

app = FastAPI(
    title="RAG Chat API",
    description="한국어 RAG 채팅 서비스 백엔드 가이드 템플릿",
    version="0.1.0",
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
