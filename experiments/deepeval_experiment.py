import os
from typing import List

from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from se_eval_eval.evaluation import MetricExperimentBase
from se_eval_eval.schema import Analysis, EvaluationResult

"""
Implements metrics from the DeepEval framework.

Resources:
- [Main Documentation](https://documentation.confident-ai.com/)
- [Repo](https://github.com/confident-ai/deepeval)
- [Summary Metric](https://github.com/confident-ai/deepeval/tree/main/deepeval/metrics/summarization)
"""

class DeepEvalFaithfulnessExperiment(MetricExperimentBase):

    METRIC_NAME = "deep_eval_faithfulness"

    @staticmethod
    def run_eval(analysis: Analysis, notice_text: str, notice_path: str) -> EvaluationResult | List[EvaluationResult]:
        model = OllamaModel("llama3.1:8b")
        results = []
        text_to_evaluate = ""
        for item in analysis.analysis_results:
            text_to_evaluate += item.reason
        metric = FaithfulnessMetric(
            model=model,
            truths_extraction_limit=10
        )
        test_case = LLMTestCase(
            input="",
            retrieval_context=[notice_text],
            actual_output=text_to_evaluate,
        )
        metric.measure(test_case)
        results.append(EvaluationResult(
            metric_name=DeepEvalFaithfulnessExperiment.METRIC_NAME,
            score=metric.score,
            reason=metric.reason,
            llm_model_name=model.model_name
        ))
        return results
