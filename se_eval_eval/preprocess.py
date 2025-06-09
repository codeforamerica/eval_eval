import asyncio
from typing import List

import ollama

from se_eval_eval.logger import logger
from se_eval_eval.utility import hydrate_document_manifest
from se_eval_eval.schema import Document, Translation, SUPPORTED_LANGUAGES
from se_eval_eval.prompts import translation

"""
Utilities for performing initial text extraction and translation of text or PDF sources.
"""


TRANSLATION_MODEL = "aya-expanse:8b"

def preprocess_manifest(manifest_path: str) -> List[Document]:
    logger.info(f"Processing manifest: {manifest_path}")
    documents = hydrate_document_manifest(manifest_path)
    logger.info("Successfully hydrated manifest")
    ollama.generate(TRANSLATION_MODEL, "")
    for document in documents:
        logger.info(f"Translating: {document.name}")
        translations = asyncio.run(translate_document(document))
        document.translations.extend(translations)
        logger.info(f"Translation complete!")
    return documents


async def translate_document(document: Document):
    to_languages = SUPPORTED_LANGUAGES
    to_languages.remove("English")
    english_translation = document.get_translation_by_language("English")
    tasks = []
    for language in to_languages:
        tasks.append(translate_text(english_translation.text, "English", language))
    return await asyncio.gather(*tasks)


async def translate_text(text, from_language, to_language) -> Translation:
    prompt = translation.baseline_translation_prompt(from_language, to_language, text)
    ret = await ollama.AsyncClient().generate(TRANSLATION_MODEL, prompt, format=Translation.model_json_schema(),
                                              stream=False)
    response = Translation.model_validate_json(ret.response)
    response.author = TRANSLATION_MODEL
    return response
