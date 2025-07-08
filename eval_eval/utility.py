import json

from eval_eval.schema import Document, Manifest

"""
Reusable utilities for preprocessing and evaluation.
"""


def hydrate_document_manifest(manifest_path: str) -> Manifest:
    # @todo assert manifest exists
    with open(manifest_path, "r") as mf:
        manifest = json.loads(mf.read())
    document_models = []
    for document in manifest["documents"]:
        if "text" not in document.keys() or document["text"] is None:
            with open(document["path"], "r") as fp:
                document["text"] = fp.read()
        document_model = Document.model_validate(document)
        document_models.append(document_model)
    return Manifest(documents=document_models)
