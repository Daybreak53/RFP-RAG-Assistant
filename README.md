# RFP-RAG-Assistant
📃 [3팀] 공공입찰 제안요청서(RFP) 분석 및 핵심 정보 추출을 위한 RAG 시스템 구축 프로젝트 

```
RFP-RAG-Assistant/
├── .github/                 
├── frontend/                # React (Vite)
├── backend/                 # FastAPI
│   ├── .env                 # 환경변수
│   ├── requirements.txt     # 의존성 패키지
│   ├── main.py              # FastAPI 실행 진입점
│   └── app/
│       ├── api/             # API 라우터 (엔드포인트: /chat 등)
│       ├── core/            # 환경설정(config.py), 예외 처리
│       ├── schemas/         # Pydantic 모델
│       └── services/        # 비즈니스 로직 (아래 파일은 예시)
│           ├── loader.py    # 데이터 파싱 및 청킹
│           ├── retriever.py # BM25, Vector, Hybrid 검색기
│           ├── generator.py # LLM 프롬프팅 및 응답 생성
│           ├── cache.py     # 시맨틱 캐싱 및 비용 절감 라우팅
│           └── evaluator.py # 평가 및 호출 횟수/비용 트래킹
│
├── .gitignore               
├── README.md                
└── run.sh                   # 프론트/백엔드 통합 실행 스크립트