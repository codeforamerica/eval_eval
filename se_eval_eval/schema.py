from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_serializer
from pydantic.json_schema import SkipJsonSchema

"""
Pydantic models for LLM structured output and data validation.
"""

class AnalysisResult(BaseModel):
    question: str = Field(description="The question being answered")
    answer: Literal["Yes", "No", "IDK"] = Field(description="Your answer")
    reason: str = Field(description="The explanation for your answer")

class Analysis(BaseModel):
    results: List[AnalysisResult]
    llm_model: SkipJsonSchema[Optional[str]] = Field(
        default="",
    )

class Document(BaseModel):
    path: str
    text: str
    notes: Optional[str] = Field(
        default=None,
    )
    analysis: Optional[List[Analysis]] = Field(
        default=[],
    )

class Manifest(BaseModel):
    documents: Optional[List[Document]] = Field(
        default=[],
    )

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
    model_name: Optional[str] = Field(
        description="The name of the model used for the evaluation."
    )
