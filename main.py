import argparse

import ollama
from dotenv import load_dotenv

from eval_eval.analysis import generate_analysis_from_manifest
from eval_eval.evaluation import run_experiments_from_manifest
from eval_eval.logger import logger
from eval_eval.utility import hydrate_document_manifest

"""
Main entrypoint script for interacting with the repo.

Run .venv/bin/python main.py --help for more information.
"""

load_dotenv()

# Run the analysis process taking a manifest.json file and returning a modified version.
CMD_ANALYZE = "analyze"
# Run the evaluation process specifying individual metrics or as an entire suite.
CMD_EVALUATE = "evaluate"
# A list of supported Ollama models that should be downloaded for complete repository usage.
SUPPORTED_OLLAMA_MODELS = ["llama3.1:8b", "qwen3:8b"]


def handle_process(args: argparse.Namespace) -> None:
    """
    Runs either the "analyze" or "evaluate" command.

    Parameters
    ----------
    args: argparse.Namespace
      Args provided from the CLI.
    """
    logger.info(f"Processing manifest: {args.manifest_path}")
    hydrated_manifest = hydrate_document_manifest(args.manifest_path)
    logger.info("Successfully hydrated manifest")
    if args.cmd == CMD_ANALYZE:
        assert_ollama_models_installed()
        if args.metrics is not None:
            raise ValueError(
                "The --metrics option cannot be used with the analyze command."
            )
        generate_analysis_from_manifest(hydrated_manifest, SUPPORTED_OLLAMA_MODELS)
        if args.output_path is not None:
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(hydrated_manifest.model_dump_json())
        else:
            print(hydrated_manifest.model_dump_json())
    if args.cmd == CMD_EVALUATE:
        metrics = []
        if args.metrics is not None:
            metrics = args.metrics.split(",")
        run_experiments_from_manifest(hydrated_manifest, metrics)
        if args.output_path is not None:
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(hydrated_manifest.model_dump_json())
        else:
            print(hydrated_manifest.model_dump_json())


def assert_ollama_models_installed():
    installed_model_names = []
    for model in ollama.list().models:
        installed_model_names.append(model.model)
    missing_models = set(SUPPORTED_OLLAMA_MODELS) - set(installed_model_names)
    if len(missing_models) > 0:
        commands = ""
        for model in missing_models:
            commands += f"ollama pull {model}\n"
        raise RuntimeError(
            f"Some required Ollama models are missing. Please run: \n\n{commands}"
        )


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
        choices=[CMD_ANALYZE, CMD_EVALUATE],
        help=f"Runs a {CMD_ANALYZE} or {CMD_EVALUATE} command that manipulates a manifest json file.",
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
