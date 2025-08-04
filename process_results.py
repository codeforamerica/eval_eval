import argparse
import os
import glob

import pandas as pd

from eval_eval.schema import Manifest
from eval_eval.utility import hydrate_document_manifest

METRIC_VALUE_RANGE = {
    "deep_eval_faithfulness": {
        "positive": 1,
        "negative": 0,
    },
    "deep_eval_answer_relevancy": {
        "positive": 1,
        "negative": 0,
    },
    "deep_eval_g_eval:clarity": {
        "positive": 1,
        "negative": 0,
    },
    "deep_eval_g_eval:bias": {
        "positive": 1,
        "negative": 0,
    },
    "deep_eval_g_eval:completeness": {
        "positive": 1,
        "negative": 0,
    },
    "deep_eval_g_eval:summarization_correctness": {
        "positive": 1,
        "negative": 0,
    },
    "mlflow_faithfulness": {
        "positive": 5,
        "negative": 1,
    },
    "opik_eval_hallucination": {
        "positive": 0,
        "negative": 1,
    },
    "promptfoo_faithfulness": {
        "positive": 1,
        "negative": 0,
    },
    "ragas_faithfulness": {
        "positive": 1,
        "negative": 0,
    }
}

QUESTION_MAP = {
    "Required Actions": "Required Actions",
    "**Required Actions**": "Required Actions",
    "What actions are required by the recipient?": "Required Actions",
    "**Required Actions**: What specific actions, if any, must the recipient take after receiving this notice? Include deadlines and consequences of inaction.": "Required Actions",
    "Is the document primarily informational or is action required?": "Document Classification",
    "**Document Classification**: Determine whether this document is primarily informational (notifying the recipient of status or updates) or action-required (demanding a response to maintain benefits).": "Document Classification",
    "**Document Classification**": "Document Classification",
    "**Document Classification**: Determine whether this document is primarily informational (notifying the recipient of status or updates) or action-required (demanding a response to maintain benefits). Explain the potential consequences if the recipient does not respond to or act on this document.": "Document Classification",
    "**Plain Language Assessment**: Evaluate whether this notice uses plain language appropriate for a 6th-grade reading level.": "Plain Language",
    "**Plain Language Assessment**: Evaluate whether this notice uses plain language appropriate for a 6th-grade reading level. Consider vocabulary complexity, sentence structure, and use of jargon or technical terms.": "Plain Language",
    "Is this notice written in plain language, at 6th-grade reading level or lower?": "Plain Language",
    "**Plain Language Assessment**": "Plain Language",
    "Plain Language Assessment": "Plain Language",
    "How could this document be more effective for the recipient?": "Effectiveness",
    "Effectiveness Improvements": "Effectiveness",
    "**Effectiveness Improvements**: Identify the most significant changes that would make this document more effective for the recipient, focusing on clarity, accessibility, and actionability.": "Effectiveness",
    "**Effectiveness Improvements**": "Effectiveness",
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifests",
        type=str,
    )
    parser.add_argument(
        "output",
        type=str,
    )
    return parser.parse_args()


def get_manifests(manifest_path: str) -> list:
    all_manifests = []
    for manifest_path in glob.glob(f"{manifest_path}"):
        all_manifests.append(manifest_path)
    return all_manifests


def process_results(manifests: list) -> pd.DataFrame:
    rows = []
    for manifest_path in manifests:
        manifest: Manifest = hydrate_document_manifest(manifest_path)
        for document in manifest.documents:
            for analysis in document.notice_analysis:
                for evaluation_result in analysis.evaluation_results:
                    value_range = METRIC_VALUE_RANGE[evaluation_result.metric_name]
                    rows.append({
                        "document": os.path.basename(document.path),
                        "analysis_llm": analysis.llm_model_name,
                        "analysis_prompt": analysis.prompt_name,
                        **evaluation_result.model_dump(),
                        **value_range
                    })
    df = pd.DataFrame(rows)
    df = df.rename(columns={"llm_model_name": "evaluation_llm"})
    df["related_analysis"] = df["related_analysis"].replace(QUESTION_MAP)
    df["related_analysis"] = df["related_analysis"].str.lower()
    df = df.drop(columns=["details", "duration"])
    print(df["related_analysis"].value_counts(dropna=False))
    return df

if __name__ == "__main__":
    provided_args = get_args()
    manifests = get_manifests(provided_args.manifests)
    results = process_results(manifests)
    results.to_csv(provided_args.output, index=False)
