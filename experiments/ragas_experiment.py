from typing import List

from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from eval_eval.evaluation import MetricExperimentBase
from eval_eval.logger import logger
from eval_eval.schema import Analysis, EvaluationResult

MODEL_NAME = "gpt-4.1"

open_ai_model = ChatOpenAI(model=MODEL_NAME)
ragas_llm = LangchainLLMWrapper(langchain_llm=open_ai_model)


class RagasFaithfulnessExperiment(MetricExperimentBase):
    METRIC_NAME = "ragas_faithfulness"

    @staticmethod
    def run_eval(
        analysis: Analysis, notice_text: str, notice_path: str
    ) -> EvaluationResult | List[EvaluationResult]:
        """
        Run RAGAS faithfulness evaluation on a given scenario.
        """
        results = []
        logger.info(f"Ragas Faithfulness: evaluating summary")
        sample = SingleTurnSample(
            user_input="Write a 2-3 sentence summary of the notice.",
            response=analysis.summary,
            retrieved_contexts=[notice_text],
        )
        scorer = Faithfulness(llm=ragas_llm)
        score = scorer.single_turn_score(sample)
        results.append(
            EvaluationResult(
                metric_name=RagasFaithfulnessExperiment.METRIC_NAME,
                score=score,
                llm_model_name=MODEL_NAME,
                related_analysis="summary",
            )
        )
        for i, q in enumerate(analysis.questions):
            logger.info(
                f"Ragas Faithfulness: evaluating question {i + 1} of {len(analysis.questions)}"
            )
            sample = SingleTurnSample(
                user_input=q.question,
                response=q.answer,
                retrieved_contexts=[notice_text],
            )
            scorer = Faithfulness(llm=ragas_llm)
            score = scorer.single_turn_score(sample)
            results.append(
                EvaluationResult(
                    metric_name=RagasFaithfulnessExperiment.METRIC_NAME,
                    score=score,
                    llm_model_name=MODEL_NAME,
                    related_analysis=q.question,
                )
            )
        return results
