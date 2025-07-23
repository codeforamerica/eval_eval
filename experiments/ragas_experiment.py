from datasets import Dataset

from typing import List

from langchain_community.llms import Ollama
from ragas.llms import LangchainLLM
from ragas import RAGAS_LLM
from ragas.metrics import faithfulness
from ragas.evaluation import evaluate


from eval_eval.evaluation import MetricExperimentBase
from eval_eval.schema import Analysis, EvaluationResult


ollama_llm = Ollama(model="deepseek-r1:8b")  # or "mistral" if you want something faster
ragas_llm = LangchainLLM(llm=ollama_llm)
RAGAS_LLM.set(ragas_llm)


class RagasFaithfulnessExperiment(MetricExperimentBase):

    METRIC_NAME = "ragas_faithfulness"
    MODEL_NAME = "deepseek-r1:8b"  

    @staticmethod
    def run_eval(
        analysis: Analysis, notice_text: str, notice_path: str
    ) -> EvaluationResult | List[EvaluationResult]:
        """
        Run RAGAS faithfulness evaluation on a given scenario.
        """
        print(f"\nSummary: {analysis.summary}\n")
        for question in analysis.questions:
            print(f"Question: {question.question}\nAnswer: {question.answer}\n")
        print("\n\n")


        if not hasattr(analysis.questions[0], "context_chunks") or not analysis.questions[0].context_chunks:
            return EvaluationResult(
                metric_name=RagasFaithfulnessExperiment.METRIC_NAME,
                score=None,
                reason="Missing context chunks; required for Ragas faithfulness evaluation.",
                llm_model_name=RagasFaithfulnessExperiment.MODEL_NAME,
            )

        # Prepare dataset for Ragas
        data = []
        for q in analysis.questions:
            data.append({
                "question": q.question,
                "answer": q.answer,
                "context": q.context_chunks,
                "ground_truth": "",  # Not required for faithfulness
            })

        ragas_dataset = Dataset.from_list(data)

        # Configure Ragas to use Ollama
        ollama_llm = Ollama(model=RagasFaithfulnessExperiment.MODEL_NAME)
        ragas_llm = LangchainLLM(llm=ollama_llm)
        RAGAS_LLM.set(ragas_llm)

        # Run the evaluation
        # Evaluate
        result = evaluate(dataset=ragas_dataset, metrics=[faithfulness])
        score = result["faithfulness"]

        return EvaluationResult(
            metric_name=RagasFaithfulnessExperiment.METRIC_NAME,
            score=score,
            reason="Evaluated with Ragas faithfulness metric using Ollama.",
            llm_model_name=RagasFaithfulnessExperiment.MODEL_NAME,
        )
