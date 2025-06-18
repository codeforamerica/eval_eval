from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_serializer
from pydantic.json_schema import SkipJsonSchema

"""
Pydantic models for LLM structured output and data validation.
"""

SUPPORTED_LANGUAGES = [
    "English",
    "Spanish",
    "Chinese",
    "Tagalog",
    "Vietnamese",
    "Russian",
    "Korean",
    "Lao",
    "Panjabi",
]


class Translation(BaseModel):
    part: SkipJsonSchema[Optional[str | int | float]] = Field(
        default=0,
        description="An identifier that links the same subset of document text across multiple languages.",
    )
    text: str = Field(description="Text to be translated.")
    language: SkipJsonSchema[Literal[*SUPPORTED_LANGUAGES]] = Field(
        default=None,
        description="The language the text should be translated into. Must be one of se_eval_eval.SUPPORTED_LANGUAGES.",
    )
    author: SkipJsonSchema[Optional[str]] = Field(
        default=None,
        description="The author of the translation or baseline for human translations.",
    )
    prompt: SkipJsonSchema[Optional[str]] = Field(
        default=None,
        description="The prompt responsible for generating the translation if created by an LLM.",
    )


class Document(BaseModel):
    en_name: str = Field(
        description="The name of the English document used for descriptive purposes.",
    )
    en_file_name: str = Field(
        description="The file name of the English document for descriptive purposes.",
    )
    en_url: str = Field(
        description="The url of the English document for descriptive purposes.",
    )
    translations: List[Translation] = Field(
        description="The list of translations associated with the document."
    )

    def get_translation_by_language(self, language: str) -> List[Translation]:
        """
        Gets a list of translations associated with the given language.

        Parameters
        ----------
        language: str
            One of the languages available for the document.

        Returns
        -------
        translations: List[Translation]
          A list of translations associated with the given language.
        """
        translation = list(
            filter(lambda x: x.language.lower() == language.lower(), self.translations)
        )
        if len(translation) == 0:
            raise RuntimeError(f"Document, {self.name} has no {language} translation.")
        return translation


class Result(BaseModel):
    metric_name: str = Field(
        description="The name of the metric used for evaluating the scenario."
    )
    score: float = Field(
        description="A score produced by the metric.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="A justification for the score.",
    )
    details: Optional[dict] = Field(
        default=None,
        description="Additional information produced by the metric to be serialized as JSON.",
    )


class Scenario(BaseModel):
    name: str = Field(
        description="The name of the scenario.",
    )
    baseline_translation: Translation = Field(
        description="The baseline or truth translation to compare against."
    )
    evaluation_translation: Translation = Field(
        description="The experimental translation."
    )
    results: List[Result] = Field(
        default=[], description="The results of the scenario from all the experiments."
    )

    def add_result(self, values: dict) -> None:
        """
        Adds a result to a scenario.

        Parameters
        ----------
        values: dict
          A list of result values, see se_eval_eval.schema.Result.
        """
        result = Result.model_validate(values)
        self.results.append(result)
