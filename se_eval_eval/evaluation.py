from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from importlib import import_module
import inspect

from se_eval_eval.logger import logger
from se_eval_eval.utility import hydrate_document_manifest
from se_eval_eval.schema import Document, Result, Scenario, SUPPORTED_LANGUAGES

"""
Utilities for running evaluation experiments.
"""

EVALUATION_BATCH_SIZE = 2


class EvalExperimentBase(ABC):
    EXPERIMENT_NAME = ""

    @staticmethod
    @abstractmethod
    def run_eval(subject: Scenario) -> Result | List[Result]:
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


def run_experiments_from_manifest(hydrated_manifest: List[Document], ai_author: str, **kwargs) -> List[Result]:
    eval_languages = SUPPORTED_LANGUAGES
    eval_languages.remove("English")
    experiment_classes = get_experiments(kwargs.get("experiment_path", "experiments"))
    results = []
    for document in hydrated_manifest:
        english_translation = document.get_translation_by_language("English").pop()
        for language in eval_languages:
            translations = document.get_translation_by_language(language)
            subject = Scenario(
                label=get_scenario_label(document, "English", language),
                baseline_translation=english_translation,
                golden_translation=next((x for x in translations if x.author == "baseline")),
                ai_translation=next((x for x in translations if x.author == ai_author)),
            )
            for experiment in experiment_classes:
                logger.info(f"Beginning Experiment: {experiment.EXPERIMENT_NAME} on {document.name} (English > {language})")
                result = experiment.run_eval(subject)
                if type(result) != list:
                    result = [result]
                results.extend(result)
    return results

def get_scenario_label(document: Document, from_label: str, to_label: str) -> str:
    document_name = document.name.lower().replace(" ", "_")
    return f"{document_name}:{from_label.lower()}:{to_label.lower()}"
