from typing import Optional, Literal, List, Any

from pydantic import BaseModel

"""
Pydantic models for LLM structured output and data validation.
"""

SUPPORTED_LANGUAGES = ["English", "Chinese", "Tagalog", "Vietnamese"]

class Translation(BaseModel):
    text: str
    language: Literal[*SUPPORTED_LANGUAGES]
    author: Optional[str]


class Document(BaseModel):
    name: str
    translations: List[Translation]

    def get_translation_by_language(self, language: str):
        translation = next((x for x in self.translations if x.language == language), None)
        if translation is None:
            raise RuntimeError(f"Document, {self.name} has no {language} translation.")
        return translation


class Result(BaseModel):
    metric_name: str
    score: float
    reason: Optional[str] = None
    details: Optional[dict] = None
    inference_model_name: Optional[str] = None
    evaluation_model_name: Optional[str] = None


class ExperimentSubject(BaseModel):
    baseline_translation: Translation
    golden_translation: Translation
    ai_translation: Translation

