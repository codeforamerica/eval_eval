import argparse

from dotenv import load_dotenv

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
        if args.experiments is not None:
            raise ValueError(
                "The --experiments option cannot be used with the translation command."
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
        experiments = []
        experiments_label = "All"
        if args.experiments is not None:
            experiments_label = args.experiments
            experiments = args.experiments.split(",")
        logger.info(f"Evaluating experiments: {experiments_label}")
        results = run_experiments_from_manifest(hydrated_manifest, experiments)
        json = model_list_to_json(results)
        if args.output_path is not None:
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(json)
        else:
            print(json)


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
        "--experiments",
        type=str,
        help="A comma separated list of experiment names to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    provided_args = get_args()
    handle_process(provided_args)
