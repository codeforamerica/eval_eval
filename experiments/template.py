import os
from typing import List

from se_eval_eval.evaluation import MetricExperimentBase
from se_eval_eval.schema import Analysis, EvaluationResult

"""
This is a space to put experiments for a given service or evaluation technique.

Add as many experiment classes as you'd like. Each implementation should extend MetricExperimentBase.
They are called automatically through the main.py script.

The run_eval method is passed a scenario, which contains an approved "baseline" and an "evaluation", ai translation to compare it against.
"""

# Uncomment this class and rename it for your service or evaluation technique.
# class YourExperiment(MetricExperimentBase):
#
#     METRIC_NAME = "your_experiment_name"
#
#     @staticmethod
#     def run_eval(scenario: Scenario):
#         """
#         Run evaluation on a given scenario.
#
#         Parameters
#         ----------
#         scenario : Scenario
#             The scenario object containing different translations (baseline_translation, evaluation_translation)
#             and a part for identification. See se_eval_eval.schema.Scenario for details.
#         """
#         scenario.add_result({
#             "metric_name": YourExperiment.METRIC_NAME,
#             "score": 0,
#             "reason": "",
#             "details": {},
#         })
#         pass
