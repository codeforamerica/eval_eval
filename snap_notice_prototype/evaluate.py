import json

from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric, FaithfulnessMetric

from generate import ANALYSIS_PROMPT

EVALUATION_MODEL = "deepseek-r1:8b"
model = OllamaModel(EVALUATION_MODEL)

def deep_eval_faithfulness(source_text: str, text_to_evaluate: str) -> list:
    results = []
    metric = FaithfulnessMetric(
        model=model,
        truths_extraction_limit=10
    )
    test_case = LLMTestCase(
        input="",
        retrieval_context=[source_text],
        actual_output=text_to_evaluate,
    )
    metric.measure(test_case)
    results.append({"metric": "deepeval_faithfulness", "score": metric.score, "reason": metric.reason})
    return results


def deep_eval_prompt_alignment(source_text: str, text_to_evaluate: str) -> list:
    results = []
    metric = PromptAlignmentMetric(
        model=model,
        prompt_instructions=[
            '"No" answers represent information that is clearly absent?',
            '"IDK" answers represent information that is unclear or ambiguous?'
            "The plain language question consider the overall readability, not just individual elements?",
            "The summary question should describe overall alignment with all previous questions."
        ]
    )
    test_case = LLMTestCase(
        input=EVALUATION_MODEL.format(notice=source_text),
        actual_output=text_to_evaluate,
    )
    metric.measure(test_case)
    results.append({"metric": "deepeval_prompt_alignment", "score": metric.score, "reason": metric.reason})
    return results


if __name__ == "__main__":
    with open("notice_analysis.json", "r") as notice_text:
        analysis_json = json.load(notice_text)
    results = []
    for analysis in analysis_json:
        with open(analysis["file"], "r") as file:
            text = file.read()
        faithfulness_result = deep_eval_faithfulness(text, analysis["response"])
        prompt_alignment_result = deep_eval_prompt_alignment(text, analysis["response"])
        results.append({"file": analysis["file"], "result": faithfulness_result + prompt_alignment_result})
    with open("evaluation_results.json", "w") as result_output:
        result_output.write(json.dumps(results))
