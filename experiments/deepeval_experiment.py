from se_eval_eval.evaluation import EvalExperimentBase, Result

class DeepEvalExperiment(EvalExperimentBase):

    metric_name = "deep_eval_experiment"

    @staticmethod
    def run_eval() -> Result:
        return Result(metric_name=DeepEvalExperiment.metric_name, score=1)
