import os
from typing import List

from se_eval_eval.evaluation import EvalExperimentBase
from se_eval_eval.schema import Scenario, Result

"""
This is a space to put experiments for a given service or evaluation technique.

Add as many experiment classes as you'd like. Each implementation should extend EvalExperimentBase.
They are called automatically through the main.py script.

The run_eval method is passed a scenario, which contains an approved "golden_translation" and an ai translation to compare it against.
"""

# Uncomment this class and rename it for your service or evaluation technique.
# class YourExperiment(EvalExperimentBase):
#
#     EXPERIMENT_NAME = "your_experiment_name"
#
#     @staticmethod
#     def run_eval(scenario: Scenario) -> Result | List[Result]:
#         """Run evaluation on a given scenario.
#
#         Parameters
#         ----------
#         scenario : Scenario
#             The scenario object containing different translations (baseline_translation, golden_translation, ai_translation)
#             and a part for identification. See se_eval_eval.schema.Scenario for details.
#
#         Returns
#         -------
#         Result or list of Result
#             The evaluation result(s) for the provided scenario.
#             See se_eval_eval.schema.Result
#         """
#         pass
