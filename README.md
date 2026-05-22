# 🤖 RFP-RAG Assistant

> **공공입찰 제안요청서(RFP) 분석 및 핵심 정보 추출을 위한 사내 RAG 시스템** > 본 프로젝트는 공공입찰 컨설팅 스타트업 **'입찰메이트'**를 위한 시스템으로, 매일 쏟아지는 수백 건의 나라장터 RFP 문서를 일일이 분석하는 대신 자연어 질의를 통해 핵심 조건·예산·일정 요구사항을 즉시 추출할 수 있도록 돕습니다. ---

## 🌟 주요 기능 (Key Features)

1. **다중 포맷 문서 파싱 및 메타데이터 자동 매칭**
   * PDF 및 HWP 양식의 제안요청서(RFP) 문서를 정확하게 파싱하고, 대조군인 CSV 메타데이터(공고 정보 등)와 매칭하여 문서별 컨텍스트 관계를 정밀하게 연결합니다.

2. **전략적 다중 청킹 기법 (Chunking Strategies) 제공**
   * 문서의 구조적 특성에 맞게 최적의 텍스트 분할을 수행할 수 있도록 **Recursive(재귀적)**, **Semantic(의미론적)**, **Sentence(문장 단위)** 3가지 청킹 알고리즘을 지원합니다.

3. **Qdrant Cloud 기반 하이브리드 벡터 DB 구축**
   * **Dense(밀집)** 및 **Sparse(희소)** 벡터 임베딩을 결합한 하이브리드 벡터DB를 Qdrant Cloud에 구현하였으며, **Payload 인덱싱**을 적용하여 대용량 데이터에서 필터링이 가능합니다.

4. **고도화된 다중 리트리버 알고리즘 (Retrieval)**
   * 질문의 의도에 맞춰 최적의 컨텍스트를 추출할 수 있도록 **Vector(벡터)**, **Keyword(키워드)**, **Hybrid(하이브리드)** 검색뿐만 아니라 중복을 제거하는 **MMR**, 가상 답변을 활용하는 **HyDE** 기법을 모두 지원합니다.

5. **검색 성능 향상을 위한 Multi-Query Retrieval 기법**
   * 사용자의 모호하거나 단답형인 질의를 LLM을 통해 다각도로 확장(Multi-Query) 생성하여 검색의 재현율(Recall)을 극대화합니다.

6. **자연어 질의 기반 자동 메타데이터 필터링 (Auto Filtering)**
   * 사용자가 입력한 자연어 질문 속에서 필터링 조건을 자동으로 추출하고, 이를 Qdrant DB의 메타데이터 필터와 실시간 연동하여 검색 대상 문서를 명확하게 타겟팅합니다.

7. **다중 LLM 및 3가지 프롬프트 엔지니어링 전략 지원**
   * **모델 선택:** 보안 및 비용 효율을 위한 로컬 오픈소스 모델(**EXAONE 3.5:7.8b**)과 다중 문서 비교에 유리한 상용 API(**gpt-5-nano**) 중 선택하여 구동할 수 있습니다.
   * **프롬프트 기법:** 환각, 출력형식을 제어하는 **'기본 4규칙 프롬프트'**, 논리적 추론을 유도하는 **'Zero-shot CoT'**, 예시를 활용하는 **'Few-shot CoT'** 총 3가지 모드를 제공합니다.

8. **RAGAS + Langfuse 기반 자동화 평가 및 모니터링**
   * RAG 파이프라인의 성능을 정량화하기 위해 **RAGAS** 평가 지표를 자동 채점하고, **Langfuse**를 연동하여 비용 추적, 답변 레이턴시(속도), 시스템 호출 흐름을 상시 모니터링합니다.

9. **실험 변인 통제를 위한 config.yaml 제어 시스템**
   * 청킹 모드, 검색 기법, Top-K, 유사도 임계값(Score Threshold) 등 RAG 성능에 영향을 주는 모든 실험적 변수를 `config.yaml` 파일 하나로 중앙 관리하여 유연한 벤치마크 실험 환경을 제공합니다.

| 구분 | 기술 스택 | 비고 |
| :--- | :--- | :--- |
| **Framework** | LangChain | RAG 파이프라인 전반 및 체인 구축 |
| **Data Parser** | PyPDFLoader, olefile & zlib (HWP Custom 파싱) | 다중 문서 포맷 대응 |
| **Tokenizer** | kiwipiepy | 한국어 형태소 분석 및 Sparse 인덱스 생성용 |
| **Embedding** | BAAI/bge-m3, text-embedding-3-small | 로컬 및 OPENAI API 병행 활용 |
| **Vector DB** | Qdrant Cloud | Dense + Sparse 하이브리드, Payload 인덱싱 지원 |
| **LLM** | EXAONE 3.5:7.8b (Ollama), gpt-5-nano (OpenAI API) | 보안용 로컬 오픈소스 모델 및 다중 문서 추론용 상용 API 조합 |
| **Ops / Eval** | RAGAS, Langfuse | 비용, 레이턴시, RAG 성능 정량 평가 파이프라인 |

##  실행 절차

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (.env)
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고, 연동할 외부 서비스의 API 키를 입력합니다.
.env.example 을 보고 API키를 입력하시면 됩니다.

### 3. 데이터 적재
RFP 문서 데이터를 프로그램 root 폴더 내부에 data 디렉토리 생성 후 적재합니다.

### 4. 실행 방법
```bash
여기에 적어주세요
```

