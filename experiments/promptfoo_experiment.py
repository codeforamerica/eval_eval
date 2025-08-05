import os
import yaml
import subprocess
import json
from typing import List

# Import the necessary components from the eval_eval package
from eval_eval.evaluation import MetricExperimentBase
from eval_eval.logger import logger
from eval_eval.schema import Analysis, EvaluationResult, Manifest, Document, AnalysisQuestion


# --- Promptfoo Evaluation Class ---
class PromptfooFaithfulnessExperiment(MetricExperimentBase):
    METRIC_NAME = "promptfoo_faithfulness"
    # These files will be dynamically named based on analysis properties to avoid conflicts
    # when processing multiple analyses.
    CONFIG_FILE_BASE = "promptfooconfig"
    OUTPUT_FILE_BASE = "promptfoo_output"

    @staticmethod
    def _generate_promptfoo_config(
            analysis: Analysis, notice_text: str, model_name: str, config_output_filename: str
    ) -> dict:
        """Generates the promptfoo config dictionary for a single analysis."""
        tests = []

        # Evaluate summary
        tests.append({
            "vars": {
                "query": "Write a 2-3 sentence summary of the notice.",
                "context": notice_text,
                "prompt": analysis.summary,
            },
            "assert": [
                {
                    "type": "context-faithfulness",
                    "threshold": 0.5,  # Default threshold
                    "provider": model_name,
                }
            ],
            "description": f"Summary Faithfulness for {analysis.llm_model_name} with {analysis.prompt_name}",
            "metadata": {
                "related_analysis_part": "summary",
                "llm_model_name": analysis.llm_model_name,
                "prompt_name": analysis.prompt_name,
            }
        })

        # Evaluate questions and answers
        for i, item in enumerate(analysis.questions):
            tests.append({
                "vars": {
                    "query": item.question,
                    "context": notice_text,
                    "prompt": item.answer,
                },
                "assert": [
                    {
                        "type": "context-faithfulness",
                        "threshold": 0.5,  # Default threshold
                        "provider": model_name,
                    }
                ],
                "description": f"Question {i + 1} Faithfulness for {analysis.llm_model_name} with {analysis.prompt_name}",
                "metadata": {
                    "related_analysis_part": item.question,
                    "llm_model_name": analysis.llm_model_name,
                    "prompt_name": analysis.prompt_name,
                }
            })

        config = {
            "description": f"Faithfulness Evaluation for {analysis.llm_model_name} ({analysis.prompt_name})",
            "providers": [f"file://experiments/promptfoo_provider.py"],
            "tests": tests,
            "outputPath": config_output_filename  # Use the passed dynamic output filename
        }
        return config

    @staticmethod
    def run_eval(
            analysis: Analysis, notice_text: str, notice_path: str
    ) -> List[EvaluationResult]:
        """
        Runs the faithfulness evaluation using promptfoo for a single analysis object.
        """
        # Determine model name from analysis, fall back to a default if not present
        # model_name = analysis.llm_model_name if analysis.llm_model_name else "llama3.1:8b"
        model_name = "openai:gpt-4.1"

        # Sanitize model and prompt names for filenames
        sanitized_model_name = analysis.llm_model_name.replace(':', '_').replace('/', '_').replace('\\', '_')
        sanitized_prompt_name = analysis.prompt_name.replace(':', '_').replace('/', '_').replace('\\', '_')

        config_filename = os.path.join("experiments",
                                       f"{PromptfooFaithfulnessExperiment.CONFIG_FILE_BASE}_{sanitized_model_name}_{sanitized_prompt_name}.yaml")
        output_filename = os.path.join("experiments",
                                       f"{PromptfooFaithfulnessExperiment.OUTPUT_FILE_BASE}_{sanitized_model_name}_{sanitized_prompt_name}.json")

        logger.info(f"Generating promptfoo config for analysis by {model_name} (Prompt: {analysis.prompt_name})")
        config_data = PromptfooFaithfulnessExperiment._generate_promptfoo_config(
            analysis, notice_text, model_name, output_filename  # Pass output_filename to config generator
        )

        with open(config_filename, "w") as f:
            yaml.dump(config_data, f, sort_keys=False)

        logger.info(f"{config_filename} generated. Running promptfoo eval...")

        evaluation_results: List[EvaluationResult] = []
        try:
            command = [
                "promptfoo",
                "eval",
                "-c", config_filename,
                "--output", output_filename,
                # "--no-auto-open", # Prevent browser from opening if running in CI/headless
                # "--max-concurrent-evaluations", "1" # Limit concurrency for local Ollama
            ]

            process = subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info("promptfoo eval completed successfully.")
            logger.debug(f"promptfoo stdout:\n{process.stdout}")
            if process.stderr:
                logger.warning(f"promptfoo stderr:\n{process.stderr}")

            # Parse results from the output JSON file
            results_data = {}
            if os.path.exists(output_filename):
                with open(output_filename, "r") as f:
                    full_json_output = json.load(f)
                # os.remove(output_filename) # Clean up output file
            else:
                logger.error(f"promptfoo output file not found: {output_filename}")
                return []

            test_case_results_list = full_json_output.get("results", {}).get("results", [])

            for test_case_result in test_case_results_list:
                for assertion_result in test_case_result.get("gradingResult", {}).get("componentResults", {}):
                    if assertion_result.get("assertion", {}).get("type", None) == "context-faithfulness":
                        score = assertion_result.get("score", 0)
                        reason = assertion_result.get("reason", "No reason provided.")

                        # Extract metadata
                        related_analysis_part = test_case_result.get("metadata", {}).get(
                            "related_analysis_part", "unknown")

                        evaluation_results.append(
                            EvaluationResult(
                                metric_name=PromptfooFaithfulnessExperiment.METRIC_NAME,
                                score=score,
                                reason=reason,
                                llm_model_name=model_name,
                                related_analysis=related_analysis_part,
                            )
                        )
            return evaluation_results

        except subprocess.CalledProcessError as e:
            if e.returncode == 100:
                logger.info(f"promptfoo eval completed with exit code 100 (expected success for some cases).")
                logger.debug(f"promptfoo stdout:\n{e.stdout}")
                if e.stderr:
                    logger.warning(f"promptfoo stderr:\n{e.stderr}")

                # If exit code is 100, proceed with parsing results as it's considered a success
                results_data = {}
                if os.path.exists(output_filename):
                    with open(output_filename, "r") as f:
                        full_json_output = json.load(f)
                else:
                    logger.error(f"promptfoo output file not found after exit code 100: {output_filename}")
                    return []

                test_case_results_list = full_json_output.get("results", {}).get("results", [])

                for test_case_result in test_case_results_list:
                    for assertion_result in test_case_result.get("gradingResult", {}).get("componentResults", {}):
                        if assertion_result.get("assertion", {}).get("type", None) == "context-faithfulness":
                            score = assertion_result.get("score", 0)
                            reason = assertion_result.get("reason", "No reason provided.")

                            # Extract metadata
                            related_analysis_part = test_case_result.get("metadata", {}).get(
                                "related_analysis_part", "unknown")

                            evaluation_results.append(
                                EvaluationResult(
                                    metric_name=PromptfooFaithfulnessExperiment.METRIC_NAME,
                                    score=score,
                                    reason=reason,
                                    llm_model_name=model_name,
                                    related_analysis=related_analysis_part,
                                )
                            )
                return evaluation_results
            else:
                logger.error(f"promptfoo eval failed with unexpected exit code {e.returncode}: {e}")
                logger.error(f"Stdout: {e.stdout}")
                logger.error(f"Stderr: {e.stderr}")
                raise  # Re-raise for other actual errors
        except FileNotFoundError:
            logger.error("Error: 'promptfoo' command not found. Is promptfoo installed and in your PATH?")
            logger.error("Install with: `npm install -g promptfoo` or `brew install promptfoo`.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise
        finally:
            # Clean up the generated config file even if an error occurs
            if os.path.exists(config_filename):
                # os.remove(config_filename) # Commented out to keep file for promptfoo view
                logger.info(f"Kept {config_filename} for promptfoo view.")


# --- Main Execution Block ---
if __name__ == "__main__":
    manifest_file_path = "manifest_with_analysis.json"

    # Load the manifest JSON
    try:
        with open(manifest_file_path, "r") as f:
            manifest_data = json.load(f)

        # Hydrate the data into the Pydantic Manifest model
        hydrated_manifest = Manifest(**manifest_data)
        logger.info(f"Successfully loaded manifest from {manifest_file_path}")

    except FileNotFoundError:
        logger.error(f"Error: Manifest file not found at {manifest_file_path}")
        exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {manifest_file_path}. Check file format.")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred while loading manifest: {e}")
        exit(1)

    # Instantiate the experiment
    experiment_instance = PromptfooFaithfulnessExperiment()

    # Iterate through documents and their analyses to run the evaluation
    updated_manifest = hydrated_manifest

    logger.info("Starting promptfoo faithfulness evaluation across all analyses in the manifest...")
    for i, document in enumerate(updated_manifest.documents):
        logger.info(f"\n--- Evaluating Document: {document.path} ({i + 1}/{len(updated_manifest.documents)}) ---")
        for j, analysis in enumerate(document.notice_analysis):
            logger.info(
                f"Processing analysis by model: {analysis.llm_model_name}, prompt: {analysis.prompt_name} "
                f"({j + 1}/{len(document.notice_analysis)})"
            )

            try:
                # Call run_eval for each analysis, providing the document text
                results_for_analysis = experiment_instance.run_eval(
                    analysis=analysis,
                    notice_text=document.text,
                    notice_path=document.path
                )

                # Append the results to the analysis's evaluation_results list
                if not analysis.evaluation_results:
                    analysis.evaluation_results = []  # Ensure it's a list
                analysis.evaluation_results.extend(results_for_analysis)
                logger.info(f"Successfully evaluated analysis. Added {len(results_for_analysis)} promptfoo results.")
            except Exception as e:
                logger.error(
                    f"Failed to evaluate analysis for model {analysis.llm_model_name}, prompt {analysis.prompt_name}: {e}")
                # Continue to next analysis even if one fails

    logger.info("\n--- All evaluations completed. Final Manifest structure: ---")
    # save the updated_manifest back to a new JSON file if needed, comment out for now

    final_output_manifest_path = os.path.join("experiments", "manifest_with_promptfoo_results.json")
    with open(final_output_manifest_path, "w") as f:
        json.dump(updated_manifest.model_dump(by_alias=True, exclude_unset=True), f, indent=2)
    logger.info(f"Updated manifest saved to {final_output_manifest_path}")

    # Print a summary of results
    print("\n--- Summary of All Promptfoo Faithfulness Scores ---")
    for doc in updated_manifest.documents:
        print(f"Document: {doc.path}")
        for an in doc.notice_analysis:
            print(f"  Model: {an.llm_model_name}, Prompt: {an.prompt_name}")
            for eval_res in an.evaluation_results:
                if eval_res.metric_name == PromptfooFaithfulnessExperiment.METRIC_NAME:
                    print(f"    - {eval_res.related_analysis}: Score={eval_res.score:.2f}, Reason='{eval_res.reason}'")
    print("\nEvaluation process finished.")
