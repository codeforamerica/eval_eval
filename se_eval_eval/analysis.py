import asyncio

import ollama

from se_eval_eval.logger import logger
from se_eval_eval.prompts.analysis import prompt_1, prompt_2
from se_eval_eval.schema import Analysis, Document, Manifest

"""
Utilities for running LLM-based analysis on the notice documents.
"""

BATCH_SIZE = 2


def generate_analysis_from_manifest(manifest: Manifest, models: list) -> Manifest:
    logger.info(
        f"Beginning analysis of {len(manifest.documents)} documents with batch size {BATCH_SIZE}"
    )
    manifest_with_analysis = asyncio.run(
        a_generate_analysis_from_manifest(manifest, models)
    )
    logger.info("Analysis complete!")
    return manifest_with_analysis


async def a_generate_analysis_from_manifest(manifest: Manifest, models: list):
    tasks = {model: [] for model in models}
    for document in manifest.documents:
        for model_name in models:
            for prompt in (prompt_1, prompt_2):
                tasks[model_name].append(
                    attach_analysis_to_document(document, model_name, prompt)
                )
    output = []
    for model_tasks in tasks.values():
        for i in range(0, len(model_tasks), BATCH_SIZE):
            batch = model_tasks[i : i + BATCH_SIZE]
            results = await asyncio.gather(*batch)
            output.extend(results)
    return Manifest(documents=output)


async def attach_analysis_to_document(
    document: Document, model_name: str, prompt: callable
) -> Document:
    logger.info(f"Analyzing {document.path} with {model_name} and {prompt.__name__}")
    prompt_text = prompt(document.text)
    ret = await ollama.AsyncClient().generate(
        model_name, prompt_text, format=Analysis.model_json_schema(), stream=False
    )
    analysis = Analysis.model_validate_json(ret.response)
    analysis.llm_model_name = model_name
    analysis.prompt_name = prompt.__name__
    document.notice_analysis.append(analysis)
    return document
