import argparse

from dotenv import load_dotenv

from se_eval_eval.llm_translation import llm_add_translations
from se_eval_eval.google_translation import google_add_translations
from se_eval_eval.evaluation import run_experiments_from_manifest
from se_eval_eval.utility import model_list_to_json, hydrate_document_manifest
from se_eval_eval.logger import logger

load_dotenv()

CMD_PREPROCESS = "preprocess"
CMD_EVALUATE = "evaluate"


def handle_process(args: argparse.Namespace):
    logger.info(f"Processing manifest: {args.manifest_path}")
    hydrated_manifest = hydrate_document_manifest(args.manifest_path)
    logger.info("Successfully hydrated manifest")
    if args.cmd == CMD_PREPROCESS:
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
        results = run_experiments_from_manifest(hydrated_manifest, "aya-expanse:8b")
        json = model_list_to_json(results)
        if args.output_path is not None:
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(json)
        else:
            print(json)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, choices=[CMD_PREPROCESS, CMD_EVALUATE], help=f"Command to run {CMD_PREPROCESS} or {CMD_EVALUATE}")
    parser.add_argument('manifest_path', type=str, help='Manifest JSON file to run command with')
    parser.add_argument('--output_path', type=str, help='Where to put the output of the command')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = get_args()
    handle_process(provided_args)