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
    def __init__(self, model_name, metrics=None):
        self.model_name = model_name

        try:
            if self.model_name == "gpt-4o-mini":
                self.eval_llm = LangchainLLMWrapper(
                    ChatOpenAI(model="gpt-4o-mini")
                )
                self.eval_embeddings = LangchainEmbeddingsWrapper(
                    OpenAIEmbeddings(model="text-embedding-3-small")
                )
                
            elif self.model_name == "gemini-3.1-flash-lite":
                self.eval_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash-lite", 
                    temperature=0,
                    n=1
                ))
                self.eval_embeddings = LangchainEmbeddingsWrapper(
                    GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                )
                
            else:
                raise ValueError(f"지원하지 않는 모델입니다: '{model_name}'")
        except Exception as e:
            print(f"[오류] LLM/Embeddings 초기화 실패. API 키를 확인해주세요: {e}")
            raise

        if metrics is None:
            context_precision = ContextPrecision()
            context_recall = ContextRecall()
            faithfulness = Faithfulness()
            answer_relevancy = AnswerRelevancy()
            
            for m in [context_precision, context_recall, faithfulness, answer_relevancy]:
                m.llm = self.eval_llm
                if hasattr(m, "embeddings"):
                    m.embeddings = self.eval_embeddings

            self.metrics = [
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ]
        else:
            self.metrics = metrics

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