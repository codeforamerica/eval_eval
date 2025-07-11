from typing import List

import mlflow
from mlflow.metrics.genai import faithfulness
import pandas as pd

class MLFlowFaithfulnessExperiment(MetricExperimentBase):

    METRIC_NAME = "mlflow_faithfulness"

    @staticmethod
    def run_eval(
        analysis: Analysis, notice_text: str, notice_path: str
    ) -> EvaluationResult | List[EvaluationResult]:

      faithfulness_metric = faithfulness(model="openai:/gpt-4.1-mini")

      # MLFlow expects to evaluate "live"; that is, it expects a set of inputs and an "LLM".
      # We can simulate the LLM using a simply python function that returns the pre-computed outputs.
      #
      # Construct the input data and the input-to-ouput function
      input_data = {
        "inputs": [],
        "context": []
      }
      output_data = []

      input_data["inputs"].append("summary")
      input_data["context"].append(notice_text)
      output_data.append(analysis.summary)

      for item in analysis.questions:
        input_data["inputs"].append(item.question)
        input_data["context"].append(notice_text)
        output_data.append(item.answer)

      simulated_llm = lambda inputs: output_data

      results = []
      with mlflow.start_run() as run:
        mlflow_results = mlflow.evaluate(
          simulated_llm,
          pd.DataFrame(input_data),
          evaluators=None,
          extra_metrics=[faithfulness_metric],
          evaluator_config={
            "col_mapping": {
              "inputs": "inputs",
              "context": "context"
            }
          }
        )

      mlflow.end_run()

      outputs = mlflow_results.tables['eval_results_table'].to_dict()
      results = []
      for i in range(0, len(outputs)):
        results.append(
          EvaluationResult(
            metric_name=MLFlowFaithfulnessExperiment.METRIC_NAME,
            score=outputs['faithfulness/v1/score'][i],
            related_alanysis=outputs['outputs'][i],
            llm_model_name='openai'
          )
      )

      return results
