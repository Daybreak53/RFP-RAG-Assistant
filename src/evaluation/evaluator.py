import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 로거 설정
logger = logging.getLogger(__name__)

# 지원하는 모델 목록 정의
SUPPORTED_GEMINI_MODELS = {"gemini-3.1-flash-lite"}
SUPPORTED_OPENAI_MODELS = {"gpt-4o-mini", "gpt-5-nano", "gpt-5-mini"}


class RagasEvaluator:
    """RAG 파이프라인 성능 평가 클래스"""

    def __init__(self, model_name: str, metrics: Optional[List[Any]] = None):
        self.model_name = model_name
        
        # 기본 평가 지표 설정
        self.metrics = metrics or [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ]

        # LLM 및 임베딩 모델 초기화
        self._initialize_models()

    def _initialize_models(self) -> None:
        """선택된 모델명에 따라 적합한 LLM과 임베딩 래퍼 초기화"""
        logger.info(f"평가용 모델 초기화 진행: {self.model_name}")
        
        try:
            # Gemini 계열 모델
            if self.model_name in SUPPORTED_GEMINI_MODELS:
                llm = ChatGoogleGenerativeAI(model=self.model_name, n=1)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
            
            # OpenAI 계열 모델
            elif self.model_name in SUPPORTED_OPENAI_MODELS or self.model_name.startswith("gpt-"):
                llm = ChatOpenAI(model=self.model_name)
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            else:
                raise ValueError(f"지원하지 않는 평가 모델입니다: '{self.model_name}'")

            # RAGAS 래퍼(Wrapper) 씌우기
            self.eval_llm = LangchainLLMWrapper(llm, bypass_temperature=True)
            self.eval_embeddings = LangchainEmbeddingsWrapper(embeddings)

        except Exception as e:
            logger.error(f"LLM/Embeddings 초기화 실패. API 키 및 설정을 확인해주세요: {e}", exc_info=True)
            raise

    def run_evaluation(self, data_samples: Dict[str, Any]) -> pd.DataFrame:
        """
        포맷팅된 데이터 샘플을 바탕으로 RAGAS 평가 실행
        """
        logger.info("RAGAS 평가를 시작합니다...")
        
        try:
            dataset = Dataset.from_dict(data_samples)
            run_config = RunConfig(timeout=120.0, max_workers=3)
            
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.eval_llm,
                embeddings=self.eval_embeddings,
                raise_exceptions=False,
                run_config=run_config
            )
            
            logger.info("RAGAS 평가가 성공적으로 완료되었습니다.")
            return result.to_pandas()
            
        except Exception as e:
            logger.error(f"RAGAS 평가 수행 중 오류 발생: {e}", exc_info=True)
            raise