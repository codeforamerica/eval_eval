from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_serializer
from pydantic.json_schema import SkipJsonSchema

"""
Pydantic models for LLM structured output and data validation.
"""


class EvaluationResult(BaseModel):
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
    llm_model_name: Optional[str] = Field(
        default=None,
        description="The name of the model used for the evaluation."
    )


class AnalysisResult(BaseModel):
    question: str = Field(
        description="The question being answered"
    )
    answer: Literal["Yes", "No", "IDK"] = Field(
        description="Your answer"
    )
    reason: str = Field(
        description="The explanation for your answer"
    )


class Analysis(BaseModel):
    analysis_results: List[AnalysisResult] = Field(
        description="A list question, answer and reason objects"
    )
    llm_model_name: SkipJsonSchema[Optional[str]] = Field(
        default="",
        description="The name of the model producing the analysis"
    )
    prompt_name: SkipJsonSchema[Optional[str]] = Field(
        default="",
        description="The name or identifier of the prompt"
    )
    evaluation_results: SkipJsonSchema[Optional[List[EvaluationResult]]] = Field(
        default=[],
        description="A list of evaluation results produced by our experiments"
    )


class Document(BaseModel):
    path: str = Field(
        description="The path to the notice document"
    )
    text: str = Field(
        description="Text from the notice document"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Notes about the notice document"
    )
    notice_analysis: Optional[List[Analysis]] = Field(
        default=[],
        description="Sets of analysis performed on the document"
    )


class Manifest(BaseModel):
    documents: Optional[List[Document]] = Field(
        default=[],
        description="A list of documents"
    )
