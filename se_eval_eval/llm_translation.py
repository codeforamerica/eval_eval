import asyncio
import copy
from typing import List

import ollama

from se_eval_eval.logger import logger
from se_eval_eval.schema import Document, Translation, SUPPORTED_LANGUAGES
from se_eval_eval.prompts import translation

"""
Utilities for performing initial text extraction and translation of text or PDF sources.
"""

def llm_add_translations(hydrated_manifest: List[Document], model_name: str) -> None:
    for document in hydrated_manifest:
        logger.info(f"Translating: {document.en_name} with {model_name}")
        translations = asyncio.run(llm_translate_document(document, model_name))
        document.translations.extend(translations)
        logger.info(f"Translation complete!")


async def llm_translate_document(document: Document, model_name: str):
    to_languages = SUPPORTED_LANGUAGES.copy()
    to_languages.remove("English")
    english_translations = document.get_translation_by_language("English")
    tasks = []
    for english_translation in english_translations:
        for language in to_languages:
            tasks.append(llm_translate_text(model_name, english_translation.text, "English", language, english_translation.part))
    return await asyncio.gather(*tasks)


async def llm_translate_text(model_name: str, text, from_language, to_language, part) -> Translation:
    prompt = translation.simple_translation_prompt(from_language, to_language, text)
    ret = await ollama.AsyncClient().generate(model_name, prompt, format=Translation.model_json_schema(),
                                              stream=False)
    response = Translation.model_validate_json(ret.response)
    response.part = part
    response.author = model_name
    return response
