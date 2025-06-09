import asyncio
from abc import ABC, abstractmethod
import math
from typing import List
from pathlib import Path
from importlib import import_module
import inspect

from se_eval_eval.schema import Result
from se_eval_eval.logger import logger
from se_eval_eval.utility import hydrate_document_manifest
from se_eval_eval.schema import Document, Translation, ExperimentSubject, SUPPORTED_LANGUAGES
from se_eval_eval.prompts import translation


"""
Utilities for running evaluation experiments.
"""

EVALUATION_BATCH_SIZE = 2

class EvalExperimentBase(ABC):

    @staticmethod
    @abstractmethod
    def run_eval() -> Result|List[Result]:
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


def run_experiments_from_manifest(manifest_path: str, ai_author: str, **kwargs) -> List[Result]:
    documents = hydrate_document_manifest(manifest_path)
    eval_languages = SUPPORTED_LANGUAGES
    eval_languages.remove("English")
    experiment_classes = get_experiments(kwargs.get("experiment_path", "./experiments"))

    experiments = []
    for document in documents:
        english_translation = document.get_translation_by_language("english")
        for language in eval_languages:
            translations = document.get_translation_by_language(language)

            subject = ExperimentSubject(
                baseline_translation=english_translation,
                golden_translation=translations.next(lambda x: x.author == "baseline"),
                ai_translation=translations.next(lambda x: x.author == ai_author),
            )
            for experiment in experiment_classes:
                experiments.append([experiment, subject])

    #batch_number = math.ceil((len(to_languages) * len(documents)) / EVALUATION_BATCH_SIZE)



#async def run_document_experiments(document: Document, ai_author: str) -> List[Result]:


#async def run_experiment(document: Document, ai_author: str) -> Result:
#