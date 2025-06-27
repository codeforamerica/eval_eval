import json

import nltk
from readability import Readability
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

from generate import NOTICE_PROMPT

EVALUATION_MODEL = "deepseek-r1:8b"

nltk.download('punkt_tab')

def deterministic_evaluations(text_to_evaluate: str) -> list:
    results = []
    r = Readability(text_to_evaluate)
    result = r.flesch()
    results.append({"metric": "flesh_ease", "score": result.ease})
    dc = r.dale_chall()
    results.append({"metric": "dale_chall", "score": dc.score})
    return results

def deep_eval_evaluation(text_to_evaluate: str) -> list:
    model = OllamaModel(EVALUATION_MODEL)
    metric = PromptAlignmentMetric(
        threshold=0.7,
        model=model,
        include_reason=True,
        prompt_instructions=[
            "Include request for proof",
            "Include reason for request of proof",
            "The availability of continued benefits",
            "Answer in simple language",
        ]
    )
    test_case = LLMTestCase(
        input=NOTICE_PROMPT,
        actual_output=text_to_evaluate,
    )
    metric.measure(test_case)
    results.append({"metric": "deepeval_prompt_alignment", "score": metric.score, "reason": metric.reason})
    return results


if __name__ == "__main__":
    with open("output.json", "r") as notice_text:
        notice_json = json.load(notice_text)
    results = []
    for notice in notice_json:
        deterministic_results = deterministic_evaluations(notice)
        deep_eval_results = deep_eval_evaluation(notice)
        results.append((deterministic_results + deep_eval_results))

    with open("results.json", "w") as result_output:
        result_output.write(json.dumps(results))
