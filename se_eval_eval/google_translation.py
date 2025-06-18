import asyncio
import copy
from typing import List

import pycountry
from googletrans import Translator

from se_eval_eval.logger import logger
from se_eval_eval.schema import SUPPORTED_LANGUAGES, Document, Translation


def get_iso639_1_code(language_name):
    try:
        language = pycountry.languages.get(name=language_name)
        return language.alpha_2
    except AttributeError:
        return "ISO 639-1 code not found"


def google_add_translations(hydrated_manifest: List[Document]) -> None:
    for document in hydrated_manifest:
        logger.info(f"Translating: {document.en_name} with Google Translate")
        translations = asyncio.run(google_translate_document(document))
        document.translations.extend(translations)
        logger.info(f"Translation complete!")


async def google_translate_document(document: Document):
    to_languages = SUPPORTED_LANGUAGES.copy()
    to_languages.remove("English")
    english_translations = document.get_translation_by_language("English")
    tasks = []
    for english_translation in english_translations:
        for language in to_languages:
            tasks.append(
                google_translate_text(
                    english_translation.text,
                    "English",
                    language,
                    english_translation.part,
                )
            )
    return await asyncio.gather(*tasks)


async def google_translate_text(text, from_language, to_language, part):
    from_language_code = get_iso639_1_code(from_language)
    to_language_code = get_iso639_1_code(to_language)
    logger.info(
        f"Translating {from_language}{from_language_code} to {to_language}{to_language_code}"
    )
    async with Translator() as translator:
        result = await translator.translate(
            text, src=from_language_code, dest=to_language_code
        )
        return Translation(
            language=to_language,
            author="google_translate",
            part=part,
            text=result.text,
            prompt=None,
        )
