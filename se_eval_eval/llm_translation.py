import asyncio
from typing import List

import ollama
from tqdm.asyncio import tqdm

from se_eval_eval.logger import logger
from se_eval_eval.prompts import translation
from se_eval_eval.schema import SUPPORTED_LANGUAGES, Document, Translation

"""
Utilities for performing translation.
"""

prompts = (translation.simple_translation_prompt,)


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
            for prompt in prompts:
                prompt_text = prompt("English", language, english_translation.text)
                tasks.append(
                    llm_translate_text(
                        model_name,
                        prompt.__name__,
                        prompt_text,
                        english_translation.part,
                        language,
                    )
                )
    return await tqdm.gather(*tasks)


async def llm_translate_text(
    model_name: str, prompt_name: str, prompt_text: str, part: int, language: str
) -> Translation:
    ret = await ollama.AsyncClient().generate(
        model_name, prompt_text, format=Translation.model_json_schema(), stream=False
    )
    response = Translation.model_validate_json(ret.response)
    response.language = language
    response.part = part
    response.author = model_name
    response.prompt = prompt_name
    return response
