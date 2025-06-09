import argparse

from se_eval_eval.preprocess import preprocess_manifest
from se_eval_eval.evaluation import run_experiments_from_manifest
from se_eval_eval.utility import documents_to_json

CMD_PREPROCESS = "preprocess"
CMD_EVALUATE = "evaluate"


def handle_process(args: argparse.Namespace):
    if args.cmd == CMD_PREPROCESS:
        documents = preprocess_manifest(args.manifest_path)
        json = documents_to_json(documents)
        if args.output_path is not None:
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(json)
        else:
            print(json)
    if args.cmd == CMD_EVALUATE:
        results = run_experiments_from_manifest(args.manifest_path, "aya-expanse:8b")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, choices=[CMD_PREPROCESS, CMD_EVALUATE], help=f"Command to run {CMD_PREPROCESS} or {CMD_EVALUATE}")
    parser.add_argument('manifest_path', type=str, help='Manifest JSON file to run command with')
    parser.add_argument('--output_path', type=str, help='Where to put the output of the command')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = get_args()
    handle_process(provided_args)