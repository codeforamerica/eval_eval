from typing import List

import evaluate
import numpy as np

from se_eval_eval.evaluation import EvalExperimentBase
from se_eval_eval.schema import Scenario, Result

"""
Implements the ROUGE metric comparing two translations for overlapping character sequences.

Resources:
 - [About Rouge](https://en.wikipedia.org/wiki/ROUGE_(metric)#:~:text=The%20metrics%20compare%20an%20automatically,produced%20summary%20and%20the%20reference.)
 - [Rouge Python](https://github.com/google-research/google-research/tree/master/rouge)
"""

class RougeExperiment(EvalExperimentBase):

    EXPERIMENT_NAME = "rouge_experiment"

    @staticmethod
    def run_eval(scenario: Scenario) -> Result | List[Result]:
        metric = evaluate.load("rouge")
        metric_result = metric.compute(
            references=[scenario.golden_translation.text], predictions=[scenario.ai_translation.text]
        )
        for key, value in metric_result.items():
            if type(value) is np.float64:
                metric_result[key] = float(value)
        experiment_name = f"{RougeExperiment.EXPERIMENT_NAME}:{scenario.label}"
        return Result(metric_name=experiment_name, score=metric_result["rougeLsum"], details=metric_result)
