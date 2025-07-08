from typing import List

from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from eval_eval.evaluation import MetricExperimentBase
from eval_eval.logger import logger
from eval_eval.schema import Analysis, EvaluationResult

"""
Implements metrics from the DeepEval framework.

Resources:
- [Main Documentation](https://documentation.confident-ai.com/)
- [Repo](https://github.com/confident-ai/deepeval)
- [Faithfulness Metric](https://github.com/confident-ai/deepeval/tree/main/deepeval/metrics/faithfulness)
"""


class DeepEvalFaithfulnessExperiment(MetricExperimentBase):

    METRIC_NAME = "deep_eval_faithfulness"

    @staticmethod
    def run_eval(
        analysis: Analysis, notice_text: str, notice_path: str
    ) -> EvaluationResult | List[EvaluationResult]:
        # Add an ID to analysis parts.
        model = OllamaModel("deepseek-r1:8b")
        text_to_evaluate = [("summary", analysis.summary)]
        for item in analysis.questions:
            text_to_evaluate.append((item.question, item.answer))
        metric = FaithfulnessMetric(model=model, truths_extraction_limit=10)

        results = []
        for i, text in enumerate(text_to_evaluate):
            logger.info(f"DeepEval Faithfulness: Evaluating step {i + 1} of {len(text_to_evaluate)}")
            test_case = LLMTestCase(
                input="",
                retrieval_context=[notice_text],
                actual_output=text[1],
            )
            metric.measure(test_case)
            results.append(
                EvaluationResult(
                    metric_name=DeepEvalFaithfulnessExperiment.METRIC_NAME,
                    score=metric.score,
                    reason=metric.reason,
                    llm_model_name=model.model_name,
                    related_analysis=text[0],
                )
            )
        return results
