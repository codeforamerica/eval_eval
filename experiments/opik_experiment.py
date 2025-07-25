from typing import List

from opik.evaluation.metrics import Hallucination
from deepeval.models import OllamaModel

from eval_eval.evaluation import MetricExperimentBase
from eval_eval.logger import logger
from eval_eval.schema import Analysis, EvaluationResult

"""
Implements metrics from the Opik framework.

Resources:
- [Main Documentation](https://www.comet.com/docs/opik/
- [Repo](https://github.com/comet-ml/opik)
- [Hallucination Metric](https://www.comet.com/docs/opik/evaluation/metrics/hallucination)
"""


class OpikHallucinationExperiment(MetricExperimentBase):
    METRIC_NAME = "opik_eval_hallucination"

    @staticmethod
    def run_eval(
            analysis: Analysis, notice_text: str, notice_path: str
    ) -> EvaluationResult | List[EvaluationResult]:
        # Add an ID to analysis parts.
        model = OllamaModel("deepseek-r1:8b")
        text_to_evaluate = [("summary", analysis.summary)]
        for item in analysis.questions:
            text_to_evaluate.append((item.question, item.answer))
        metric = Hallucination()

        results = []
        for i, text in enumerate(text_to_evaluate):
            logger.info(f"Opik Hallucination: Evaluating step {i + 1} of {len(text_to_evaluate)}")
            result = metric.score(input=text[0], output=text[1], context=[notice_text])
            results.append(
                EvaluationResult(
                    metric_name=OpikHallucinationExperiment.METRIC_NAME,
                    score=result.value,
                    reason=result.reason,
                    llm_model_name=model.model_name,
                    related_analysis=text[0]
                )
            )
        return results
