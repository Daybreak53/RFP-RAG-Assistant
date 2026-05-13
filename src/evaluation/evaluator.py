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

class RagasEvaluator:
    def __init__(self, provider="gemini", metrics=None):
        self.metrics = metrics or [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ]

        self.provider = provider.lower()

        try:
            if self.provider == "openai":
                self.eval_llm = LangchainLLMWrapper(
                    ChatOpenAI(model="gpt-4o-mini")
                )
                self.eval_embeddings = LangchainEmbeddingsWrapper(
                    OpenAIEmbeddings(model="text-embedding-3-small")
                )
                
            elif self.provider == "gemini":
                self.eval_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash-lite", 
                    temperature=0,
                    n=1
                ))
                self.eval_embeddings = LangchainEmbeddingsWrapper(
                    GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                )
                
            else:
                raise ValueError(f"지원하지 않는 Provider입니다: '{provider}'")
        except Exception as e:
            print(f"[오류] LLM/Embeddings 초기화 실패. API 키를 확인해주세요: {e}")
            raise

    def run_evaluation(self, data_samples: dict):
        print("RAGAS 평가를 시작합니다...")
        
        dataset = Dataset.from_dict(data_samples)
        
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
            raise_exceptions=False
        )
        
        return result.to_pandas()