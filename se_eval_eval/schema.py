from typing import Optional, Literal, List, Any

from pydantic import BaseModel

class Translation(BaseModel):
    def __init__(self, /, **data: Any) -> None:
        if data.get("path", None) is not None:
            with open(data["path"], "r") as translation_file:
                data["text"] = translation_file.read()
                del data["path"]
        super().__init__(**data)

    text: str
    language: Literal["English", "Chinese", "Tagalog", "Vietnamese"]
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
