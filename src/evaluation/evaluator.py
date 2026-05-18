import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

class RagasEvaluator:
    def __init__(self, model_name, metrics=None):
        self.metrics = metrics or [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ]

        self.model_name = model_name

        try:
            if self.model_name == "gemini-3.1-flash-lite":
                self.eval_llm = LangchainLLMWrapper(
                    ChatGoogleGenerativeAI(
                        model="gemini-3.1-flash-lite", 
                        n=1
                    ),
                    bypass_temperature=True
                )
                self.eval_embeddings = LangchainEmbeddingsWrapper(
                    GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
                )
            elif self.model_name in ["gpt-4o-mini", "gpt-5-nano", "gpt-5-mini"] :
                self.eval_llm = LangchainLLMWrapper(
                    ChatOpenAI(model=model_name), 
                    bypass_temperature=True
                )
                self.eval_embeddings = LangchainEmbeddingsWrapper(
                    OpenAIEmbeddings(model="text-embedding-3-small"))
            else:
                raise ValueError(f"지원하지 않는 모델입니다: '{model_name}'")
        except Exception as e:
            print(f"[오류] LLM/Embeddings 초기화 실패. API 키를 확인해주세요: {e}")
            raise

    def run_evaluation(self, data_samples: dict):
        print("RAGAS 평가를 시작합니다...")
        
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
        
        return result.to_pandas()