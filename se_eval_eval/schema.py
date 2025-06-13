from typing import Optional, Literal, List, Any

from pydantic import BaseModel

"""
Pydantic models for LLM structured output and data validation.
"""

SUPPORTED_LANGUAGES = ["English", "Spanish", "Chinese", "Tagalog", "Vietnamese", "Russian", "Korean", "Lao", "Panjabi"]

class Translation(BaseModel):
    part: Optional[str|int|float]
    text: str
    language: Literal[*SUPPORTED_LANGUAGES]
    author: Optional[str]


class Document(BaseModel):
    en_name: str
    en_file_name: str
    en_url: str
    translations: List[Translation]

    def get_translation_by_language(self, language: str) -> List[Translation]:
        translation = list(filter(lambda x: x.language.lower() == language.lower(), self.translations))
        if len(translation) == 0:
            raise RuntimeError(f"Document, {self.name} has no {language} translation.")
        return translation


class Result(BaseModel):
    metric_name: str
    score: float
    reason: Optional[str] = None
    details: Optional[dict] = None
    inference_model_name: Optional[str] = None
    evaluation_model_name: Optional[str] = None


class Scenario(BaseModel):
    label: str
    baseline_translation: Translation
    golden_translation: Translation
    ai_translation: Translation
