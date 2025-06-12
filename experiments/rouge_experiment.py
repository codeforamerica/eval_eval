import json
import evaluate
import numpy as np


"""
Implements the ROUGE metric comparing two translations for overlapping character sequences.

Resources:
 - [About Rouge](https://en.wikipedia.org/wiki/ROUGE_(metric)#:~:text=The%20metrics%20compare%20an%20automatically,produced%20summary%20and%20the%20reference.)
 - [Rouge Python](https://github.com/google-research/google-research/tree/master/rouge)
"""

def load_json() -> list:
    with open("truthset.json") as f:
        return json.load(f)


def run_rouge() -> list:
    # This is pseudocode for now.
    documents = load_json()
    results = []
    for language in ("Spanish", "Chinese", "Vietnamese", "Tagalog"):
        ai_translations = list(filter(lambda x: x["language"] == language and x["author"] == "aya-expanse:8b", documents))
        for ai_translation in ai_translations:
            baseline_translation = next((for x in documents if x["language"] == language and x["author"] == "baseline" and x["part"] == ai_translation["part"]))
            assert baseline_translation is not None
            metric = evaluate.load("rouge")
            metric_result = metric.compute(
                references=[baseline_translation["text"]], predictions=[ai_translation["text"]]
            )
            for key, value in metric_result.items():
                if type(value) is np.float64:
                    metric_result[key] = float(value)
            results.append([metric_result["rougeL"], metric_result])
    return results


if __name__ == "__main__":
    run_rouge()