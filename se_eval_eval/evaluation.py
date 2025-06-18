from abc import ABC, abstractmethod
from typing import List, Literal
from pathlib import Path
from importlib import import_module
import inspect

from se_eval_eval.logger import logger
from se_eval_eval.schema import Document, Result, Scenario, SUPPORTED_LANGUAGES, Translation

"""
Utilities for running evaluation experiments.
"""

EVALUATION_BATCH_SIZE = 2

EVAL_LANGUAGES = SUPPORTED_LANGUAGES
EVAL_LANGUAGES.remove("English")


class EvalExperimentBase(ABC):
    EXPERIMENT_NAME = ""

    @staticmethod
    @abstractmethod
    def run_eval(scenario: Scenario) -> Result | List[Result]:
        pass


def get_experiments(experiment_dir: str) -> List[EvalExperimentBase]:
    experiments: List[EvalExperimentBase] = []
    for f in Path(experiment_dir).glob(f"*.py"):
        module_name = f.stem
        if (not module_name.startswith("_")) and (module_name not in globals()):
            module = import_module(f"{experiment_dir}.{module_name}")
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, EvalExperimentBase) and obj != EvalExperimentBase:
                    experiments.append(obj)
    return experiments


def run_experiments_from_manifest(hydrated_manifest: List[Document], experiments: list, **kwargs) -> List[Scenario]:
    experiment_classes = get_experiments(kwargs.get("experiment_path", "experiments"))
    if len(experiments) > 0:
        experiment_classes = list(filter(lambda x: x.EXPERIMENT_NAME in experiments, experiment_classes))

    # 1. human vs aya simple prompt
    all_scenarios = []
    for language in EVAL_LANGUAGES:
        scenarios = _run_scenario(experiment_classes, hydrated_manifest, {"language": language, "author": "baseline"}, {"language": language, "author": "aya-expanse:8b"})
        all_scenarios.extend(scenarios)
    # 2. human vs. nemo simple prompt
        scenarios = _run_scenario(experiment_classes, hydrated_manifest, {"language": language, "author": "baseline"}, {"language": language, "author": "mistral-nemo:latest"})
        all_scenarios.extend(scenarios)
    # @todo 3. human vs. aya better prompt
    # @todo 4. human vs. nemo better prompt
    return all_scenarios


def _run_scenario(experiment_classes: List[EvalExperimentBase], hydrated_manifest: List[Document], from_condition: dict, to_condition: dict) -> List[Scenario]:
    all_scenarios = []
    document_scenarios = _get_scenario(_get_scenario_name(from_condition, to_condition), hydrated_manifest, from_condition, to_condition)
    all_scenarios.extend(document_scenarios)

    for experiment_class in experiment_classes:
        for scenario in all_scenarios:
            logger.info(f"Running experiment: {experiment_class.EXPERIMENT_NAME} for scenario: {scenario.name}")
            experiment_class.run_eval(scenario)
            logger.info("Experiment complete!")
    return all_scenarios


def _filter_translations(conditions: dict) -> callable:
     def _filter(translation: Translation) -> bool:
        result = True
        keys = conditions.keys()
        if "language" in keys and translation.language != conditions["language"]:
            result = False
        if "author" in keys and translation.author != conditions["author"]:
            result = False
        if "prompt" in keys and translation.prompt != conditions["prompt"]:
            result = False
        return result
     return _filter


def _get_scenario_name(from_condition: dict, to_condition: dict) -> str:
    bits = list(from_condition.values()) + list(to_condition.values())
    bits = filter(lambda x: x != from_condition["language"], bits)
    return "_".join(bits) + "_" + from_condition["language"].lower()


def _get_scenario(name: str, manifest: List[Document], from_conditions: dict, to_conditions: dict) -> List[Scenario]:
    scenarios = []
    for document in manifest:
        # @todo assert only one value returned for each.
        from_translation = list(filter(_filter_translations(from_conditions), document.translations))
        to_translation = list(filter(_filter_translations(to_conditions), document.translations))
        if len(from_translation) == 0 or len(to_translation) == 0:
            logger.warning(f"Skipping scenario {name} cannot find required translations in manifest for {document.en_name}.")
            continue
        scenarios.append(
            Scenario(
                name= name,
                baseline_translation=from_translation.pop(),
                evaluation_translation=to_translation.pop(),
            )
        )
    return scenarios
