import inspect
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import List

from se_eval_eval.logger import logger
from se_eval_eval.schema import Analysis, Manifest, EvaluationResult

"""
Utilities for running evaluation experiments.
"""


class MetricExperimentBase(ABC):
    METRIC_NAME = ""

    @staticmethod
    @abstractmethod
    def run_eval(analysis: Analysis, notice_text: str, notice_path: str) -> EvaluationResult | List[EvaluationResult]:
        pass


def get_experiments(experiment_dir: str) -> List[MetricExperimentBase]:
    experiments: List[MetricExperimentBase] = []
    for f in Path(experiment_dir).glob(f"*.py"):
        module_name = f.stem
        if (not module_name.startswith("_")) and (module_name not in globals()):
            module = import_module(f"{experiment_dir}.{module_name}")
            for name, obj in inspect.getmembers(module):
                if (
                        inspect.isclass(obj)
                        and issubclass(obj, MetricExperimentBase)
                        and obj != MetricExperimentBase
                ):
                    experiments.append(obj)
    return experiments


def run_experiments_from_manifest(hydrated_manifest: Manifest, metrics: list, **kwargs) -> Manifest:
    experiment_classes = get_experiments(kwargs.get("experiment_path", "experiments"))
    if len(metrics) > 0:
        filtered_experiment_classes = []
        for metric in metrics:
            experiment_class = next(
                (x for x in experiment_classes if x.METRIC_NAME == metric), None
            )
            if experiment_class is None:
                raise ValueError(
                    f"Could not find provided metric {metric}. Aborting run!"
                )
            filtered_experiment_classes.append(experiment_class)
        experiment_classes = filtered_experiment_classes
        logger.info(f"Evaluating metrics: {",".join(metrics)}")
    else:
        logger.info(f"Evaluating metrics: All")
    for document in hydrated_manifest.documents:
        for analysis in document.notice_analysis:
            for experiment in experiment_classes:
                logger.info(f"Beginning: {experiment.METRIC_NAME} evaluating analysis of {document.path} produced by {analysis.llm_model_name}")
                results = experiment.run_eval(analysis, document.text, document.path)
                if type(results) is not list:
                    results = [results]
                analysis.evaluation_results.extend(results)