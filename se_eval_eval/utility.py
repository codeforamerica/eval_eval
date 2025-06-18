import json
from typing import List

from pydantic import BaseModel
from pypdf import PdfReader

from se_eval_eval.schema import Document

"""
Reusable utilities for preprocessing and evaluation.
"""


def hydrate_document_manifest(manifest_path: str):
    # @todo assert manifest exists
    with open(manifest_path, "r") as mf:
        manifest = json.loads(mf.read())
    document_models = []
    for document in manifest:
        translations = []
        for translation in document["translations"]:
            if "prompt" not in translation.keys():
                translation["prompt"] = None
            if "path" in translation.keys():
                # todo assert path exists
                if ".txt" in translation["path"]:
                    with open(translation["path"], "r") as translation_file:
                        translation["text"] = translation_file.read()
                if ".pdf" in translation["path"]:
                    reader = PdfReader(translation["path"])
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    translation["text"] = text.strip()
                del translation["path"]
                # @todo throw
            translations.append(translation)
        document_model = Document(
            en_name=document["en_name"],
            en_file_name=document["en_file_name"],
            en_url=document["en_url"],
            translations=translations,
        )
        document_models.append(document_model)
    return document_models


def model_list_to_json(models: List[BaseModel]) -> str:
    output = []
    for model in models:
        output.append(model.model_dump())
    return json.dumps(output, ensure_ascii=False, indent=2)
