import argparse

from dotenv import load_dotenv
import ollama

from se_eval_eval.evaluation import run_experiments_from_manifest
from se_eval_eval.google_translation import google_add_translations
from se_eval_eval.llm_translation import llm_add_translations
from se_eval_eval.logger import logger
from se_eval_eval.utility import hydrate_document_manifest, model_list_to_json

"""
Main entrypoint script for interacting with the repo.

Run .venv/bin/python main.py --help for more information.
"""

load_dotenv()

# Run the translation process taking a manifest.json file and returning a modified version.
# Supports initial English translation in text, txt files or PDFs.
CMD_TRANSLATE = "translate"
# Run the evaluation process specifying experiments or as an entire suite.
CMD_EVALUATE = "evaluate"
# A list of supported Ollama models that should be downloaded for complete repository usage.
SUPPORTED_OLLAMA_MODELS = ["aya-expanse:8b", "mistral-nemo:latest"]

def handle_process(args: argparse.Namespace) -> None:
    """
    Runs either the "translate" or "evaluate" command.

    Parameters
    ----------
    args: argparse.Namespace
      Args provided from the CLI.
    """
    logger.info(f"Processing manifest: {args.manifest_path}")
    hydrated_manifest = hydrate_document_manifest(args.manifest_path)
    logger.info("Successfully hydrated manifest")
    if args.cmd == CMD_TRANSLATE:
        assert_ollama_models_installed()
        if args.metrics is not None:
            raise ValueError(
                "The --metrics option cannot be used with the translation command."
            )
        llm_add_translations(hydrated_manifest, "aya-expanse:8b")
        llm_add_translations(hydrated_manifest, "mistral-nemo:latest")
        google_add_translations(hydrated_manifest)
        json = model_list_to_json(hydrated_manifest)
        if args.output_path is not None:
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(json)
        else:
            print(json)
    if args.cmd == CMD_EVALUATE:
        metrics = []
        if args.metrics is not None:
            metrics = args.metrics.split(",")
        results = run_experiments_from_manifest(hydrated_manifest, metrics)
        json = model_list_to_json(results)
        if args.output_path is not None:
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(json)
        else:
            print(json)


def assert_ollama_models_installed():
    installed_model_names = []
    logger.info(ollama.list())
    for model in ollama.list().models:
        installed_model_names.append(model.model)
    missing_models = set(SUPPORTED_OLLAMA_MODELS) - set(installed_model_names)
    if len(missing_models) > 0:
        commands = ""
        for model in missing_models:
            commands += f"ollama pull {model}\n"
        raise RuntimeError(f"Some required Ollama models are missing. Please run: \n\n{commands}")


def get_args() -> argparse.Namespace:
    """
    Gets args from the CLI.

    Returns
    -------
    args: argparse.Namespace
      Args provided from the CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cmd",
        type=str,
        choices=[CMD_TRANSLATE, CMD_EVALUATE],
        help=f"Command to run {CMD_TRANSLATE} or {CMD_EVALUATE}",
    )
    parser.add_argument(
        "manifest_path", type=str, help="Manifest JSON file to run command with"
    )
    parser.add_argument(
        "--output_path", type=str, help="Where to put the output of the command"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="A comma separated list of metric names to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    provided_args = get_args()
    handle_process(provided_args)
