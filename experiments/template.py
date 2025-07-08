from typing import List

from eval_eval.evaluation import MetricExperimentBase
from eval_eval.schema import Analysis, EvaluationResult

"""
This is a space to put experiments for a given service or evaluation technique.

Add as many experiment classes as you'd like. Each implementation should extend MetricExperimentBase.
They are called automatically through the main.py script.

The run_eval method is passed an analysis instance with summary and assessment questions.
The original document text and path are provided in case you need them.
"""


# Uncomment this class and rename it for your service or evaluation technique.
# class YourExperiment(MetricExperimentBase):
#
#     METRIC_NAME = "your_experiment_name"
#
#     @staticmethod
#     def run_eval(
#         analysis: Analysis, notice_text: str, notice_path: str
#     ) -> EvaluationResult | List[EvaluationResult]:
#         """
#         Run evaluation on a given scenario.
#
#         Parameters
#         ----------
#         analysis : Analysis
#             The analysis with summary and quality assessment questions.
#         notice_text : str
#             The original notice text.
#         notice_path : str
#             The path to the file for reference.
#         """
#         print(f"\nSummary: {analysis.summary}\n")
#         for question in analysis.questions:
#             print(f"Question: {question.question}\nAnswer: {question.answer}\n")
#         print("\n\n")
#         return EvaluationResult(
#             metric_name=YourExperiment.METRIC_NAME,
#             score=0.5,
#             reason="Optional reason for the score.",
#             llm_model_name="Optional name of llm model used.",
#         )
