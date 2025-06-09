from typing import List
from pathlib import Path
from importlib import import_module
import inspect
import json

from se_eval_eval.evaluation import EvalExperimentBase
from se_eval_eval.schema import Document


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


def hydrate_document_manifest(manifest_path: str):
    with open(manifest_path, "r") as mf:
        manifest = json.loads(mf.read())
    document_models = []
    for document in manifest:
        document_model = Document.model_validate(document)
        document_models.append(document_model)
    return document_models