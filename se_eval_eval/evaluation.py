from abc import ABC, abstractmethod
from typing import List

from se_eval_eval.schema import Result

class EvalExperimentBase(ABC):

    @staticmethod
    @abstractmethod
    def run_eval() -> Result|List[Result]:
        pass
