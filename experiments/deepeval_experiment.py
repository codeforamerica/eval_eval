from typing import List

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.models import DeepEvalBaseLLM, GPTModel, OllamaModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.tracing import observe

from eval_eval.evaluation import MetricExperimentBase
from eval_eval.logger import logger
from eval_eval.schema import Analysis, EvaluationResult

"""
Implements metrics from the DeepEval framework.

Resources:
- [Main Documentation](https://documentation.confident-ai.com/)
- [Repo](https://github.com/confident-ai/deepeval)
- [Faithfulness Metric](https://github.com/confident-ai/deepeval/tree/main/deepeval/metrics/faithfulness)
- [Answer Relevancy Metric](https://github.com/confident-ai/deepeval/tree/main/deepeval/metrics/answer_relevancy)
- [GEval](https://www.confident-ai.com/blog/g-eval-the-definitive-guide)
"""

# Deepseek is a fairly good contender from the Ollama model offerings.
# However, it is very slow compared to an OpenAI model.
EVAL_MODEL = "gpt-4.1"


def _get_model(model_name: str) -> DeepEvalBaseLLM:
    if "gpt" in model_name:
        return GPTModel(model=model_name)
    else:
        return OllamaModel(model=model_name)


class DeepEvalFaithfulnessExperiment(MetricExperimentBase):
    METRIC_NAME = "deep_eval_faithfulness"

    @staticmethod
    @observe
    def run_eval(
        analysis: Analysis, notice_text: str, notice_path: str
    ) -> EvaluationResult | List[EvaluationResult]:
        # Add an ID to analysis parts.
        model = _get_model(EVAL_MODEL)
        text_to_evaluate = [("summary", analysis.summary)]
        for item in analysis.questions:
            text_to_evaluate.append((item.question, item.answer))
        metric = FaithfulnessMetric(model=model, truths_extraction_limit=10)

        results = []
        for i, text in enumerate(text_to_evaluate):
            logger.info(
                f"DeepEval Faithfulness: Evaluating step {i + 1} of {len(text_to_evaluate)}"
            )
            test_case = LLMTestCase(
                input="",
                retrieval_context=[notice_text],
                actual_output=text[1],
            )
            metric.measure(test_case)
            logger.info(f"DeepEval reports cost of {metric.evaluation_cost}")
            results.append(
                EvaluationResult(
                    metric_name=DeepEvalFaithfulnessExperiment.METRIC_NAME,
                    score=metric.score,
                    reason=metric.reason,
                    llm_model_name=model.model_name,
                    related_analysis=text[0],
                )
            )
        return results


class DeepEvalAnswerRelevancyExperiment(MetricExperimentBase):
    METRIC_NAME = "deep_eval_answer_relevancy"

    QUESTION_MAP = {
        "Required Actions": "**Required Actions**: What specific actions, if any, must the recipient take after receiving this notice? Include deadlines and consequences of inaction.",
        "Document Classification": "**Document Classification**: Determine whether this document is primarily informational (notifying the recipient of status or updates) or action-required (demanding a response to maintain benefits). Explain the potential consequences if the recipient does not respond to or act on this document.",
        "Plain Language Assessment": "**Plain Language Assessment**: Evaluate whether this notice uses plain language appropriate for a 6th-grade reading level. Consider vocabulary complexity, sentence structure, and use of jargon or technical terms.",
        "Effectiveness Improvements": "**Effectiveness Improvements**: Identify the most significant changes that would make this document more effective for the recipient, focusing on clarity, accessibility, and actionability.",
    }

    @staticmethod
    @observe
    def run_eval(
        analysis: Analysis, notice_text: str, notice_path: str
    ) -> EvaluationResult | List[EvaluationResult]:
        # Add an ID to analysis parts.
        model = _get_model(EVAL_MODEL)
        metric = AnswerRelevancyMetric(model=model)
        results = []
        for i, item in enumerate(analysis.questions):
            logger.info(
                f"DeepEval Answer Relevancy: Evaluating step {i + 1} of {len(analysis.questions)}"
            )
            question = item.question.strip()
            if (
                analysis.prompt_name == "prompt_2"
                and question in DeepEvalAnswerRelevancyExperiment.QUESTION_MAP.keys()
            ):
                logger.info("Using expanded question for prompt_2.")
                question = DeepEvalAnswerRelevancyExperiment.QUESTION_MAP[question]
            test_case = LLMTestCase(
                input=question,
                actual_output=item.answer,
            )
            metric.measure(test_case)
            logger.info(f"DeepEval reports cost of {metric.evaluation_cost}")
            results.append(
                EvaluationResult(
                    metric_name=DeepEvalAnswerRelevancyExperiment.METRIC_NAME,
                    score=metric.score,
                    reason=metric.reason,
                    llm_model_name=model.model_name,
                    related_analysis=item.question,
                )
            )
        return results


class DeepEvalGEvalExperiment(MetricExperimentBase):
    METRIC_NAME = "deep_eval_g_eval"

    @staticmethod
    def _run_g_eval_metrics(
        metrics: List[GEval], test_case: LLMTestCase, related_analysis: str
    ) -> List[EvaluationResult]:
        results = []
        for metric in metrics:
            logger.info(f"DeepEval GEval {metric.name}: Evaluating {related_analysis}")
            metric.measure(test_case)
            logger.info(f"DeepEval reports cost of {metric.evaluation_cost}")
            results.append(
                EvaluationResult(
                    metric_name=f"{DeepEvalGEvalExperiment.METRIC_NAME}:{metric.name.lower().replace(" ", "_")}",
                    score=metric.score,
                    reason=metric.reason,
                    llm_model_name=metric.model.model_name,
                    related_analysis=related_analysis,
                )
            )
        return results

    @staticmethod
    @observe
    def run_eval(
        analysis: Analysis, notice_text: str, notice_path: str
    ) -> EvaluationResult | List[EvaluationResult]:
        # Add an ID to analysis parts.
        model = _get_model(EVAL_MODEL)
        summary_metric = GEval(
            model=model,
            name="Summarization Correctness",
            evaluation_steps=[
                "Identify whether the key facts and points in the input are preserved in the summary.",
                "Check for hallucinations — information not present in the input.",
                "Ensure the summary maintains factual consistency and avoids misrepresentation.",
                "Evaluate whether the summary omits any critical information that changes the original meaning.",
            ],
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.INPUT,
            ],
        )
        metrics = [
            GEval(
                model=model,
                name="Clarity",
                evaluation_steps=[
                    "Evaluate whether the response uses clear and direct language.",
                    "Check if the explanation avoids jargon or explains it when used.",
                    "Assess whether complex ideas are presented in a way that’s easy to follow.",
                    "Identify any vague or confusing parts that reduce understanding.",
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            ),
            GEval(
                model=model,
                name="Bias",
                strict_mode=True,
                evaluation_steps=[
                    "Identify any language that attributes characteristics, abilities, or behaviors to individuals based solely on their demographic group (race, gender, age, religion, nationality, etc.).",
                    "Check for harmful generalizations or assumptions about protected groups, including subtle implications or coded language.",
                    "Evaluate whether the output treats different demographic groups with equal respect and avoids perpetuating negative stereotypes.",
                    "Assess if the response shows unfair preference for or against certain groups in recommendations, advice, or decision-making contexts.",
                    "Check whether the output uses inclusive language and avoids assumptions about default identities (e.g., assuming 'doctor' means male).",
                    "Identify any cultural bias that privileges certain cultural perspectives while marginalizing others.",
                    "Evaluate whether historical or social context is presented fairly without reinforcing harmful narratives about marginalized groups.",
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            ),
            GEval(
                model=model,
                name="Completeness",
                evaluation_steps=[
                    "Determine if the response answers every part of the input or question.",
                    "Identify any missing elements, skipped sub-questions, or incomplete reasoning.",
                    "Check whether the output provides sufficient detail for each aspect mentioned.",
                    "Do not penalize for brevity if the coverage is complete and accurate.",
                ],
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.INPUT,
                ],
            ),
        ]
        results = []
        test_case = LLMTestCase(
            input=notice_text,
            actual_output=analysis.summary,
        )
        results.extend(
            DeepEvalGEvalExperiment._run_g_eval_metrics(
                metrics + [summary_metric], test_case, "summary"
            )
        )
        for i, item in enumerate(analysis.questions):
            test_case = LLMTestCase(
                input=notice_text,
                actual_output=item.answer,
            )
            results.extend(
                DeepEvalGEvalExperiment._run_g_eval_metrics(
                    metrics, test_case, item.question
                )
            )
        return results
