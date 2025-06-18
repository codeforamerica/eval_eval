import os
from typing import List

from deepeval.metrics import SummarizationMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from se_eval_eval.evaluation import MetricExperimentBase
from se_eval_eval.schema import Scenario

"""
Implements metrics from the DeepEval framework.

Resources:
- [Main Documentation](https://documentation.confident-ai.com/)
- [Repo](https://github.com/confident-ai/deepeval)
- [Summary Metric](https://github.com/confident-ai/deepeval/tree/main/deepeval/metrics/summarization)
"""


def _convert_model_list(list: List) -> list:
    return [dict(item) if isinstance(item, BaseModel) else item for item in list]


class DeepMetricSummaryExperiment(MetricExperimentBase):

    METRIC_NAME = "deep_eval_summarization"

    @staticmethod
    def run_eval(scenario: Scenario):
        model = GeminiModel(
            model_name="gemini-2.5-pro-preview-03-25",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
        test_case = LLMTestCase(
            input=scenario.baseline_translation.text,
            actual_output=scenario.evaluation_translation.text,
        )
        metric = SummarizationMetric(
            model=model,
        )
        metric.measure(test_case)
        details = {
            "truths": metric.truths,
            "claims": metric.claims,
            "assessment_questions": _convert_model_list(metric.assessment_questions),
            "coverage_verdicts": _convert_model_list(metric.coverage_verdicts),
            "alignment_verdicts": _convert_model_list(metric.alignment_verdicts),
        }
        scenario.add_result(
            {
                "metric_name": DeepMetricSummaryExperiment.METRIC_NAME,
                "score": metric.score,
                "reason": metric.reason,
                "details": details,
            }
        )
