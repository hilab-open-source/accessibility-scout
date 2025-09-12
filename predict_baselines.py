
import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import json
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import torch

from src.environment import Environment
from src.evaluator import EnvironmentEvaluator
from src.user_modeler import UserModeler

torch.multiprocessing.set_start_method(
    "spawn", force=True
)  # torch solution for multiprocessing

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

MAX_CONCURRENT = 8


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Environment Accessibility Evaluation Environment Cache Maker",
        description="Runs a user model on all the given baselines",
    )

    parser.add_argument(
        "-bd",
        "--baseline-dir",
        type=str,
        help="Directory of configured baselines. Each folder should be an individual environment to process",
        required=True,
    )

    parser.add_argument(
        "-um",
        "--user-model-path",
        type=str,
        help="Path to the user model to use",
        required=True
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save all the final baselines",
        required=True,
    )

    parser.add_argument(
        "--save-all",
        type=bool,
        help="Save the entire eval or just save the concerns",
        default=False
    )


    return parser


def make_baseline_task(args):

    baseline_path, save_path, user_model, save_all = args
    try:
        with HiddenPrints(): # hides all the printing
            evaluator = EnvironmentEvaluator(
                api_key=API_KEY, memory_dir=baseline_path, high_fidelity=False
            )
            evaluator.update_user_model(user_model)
            if save_all:
                evaluator.save_memory(save_path)
            else:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                out_concerns = evaluator.get_env_concerns(as_dict=True)
                with open(save_dir / "env_concerns.json", "w") as f:
                    json.dump(out_concerns, f, indent=4)

    except Exception as e:
        print(e)
        return False

    return True


def main(baseline_dir: str, user_model_path: str, save_dir: str, save_all: bool):
    baseline_dir = Path(baseline_dir)
    user_model_path = Path(user_model_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)


    with open(user_model_path, "r") as f:
        user_model = json.load(f)

    args = [(d, save_dir / d.stem, user_model, save_all) for d in baseline_dir.iterdir() if d.is_dir()]

    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        results = list(tqdm(executor.map(make_baseline_task, args), total=len(args)))

    # make_baseline_task(args[0])


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    main(args["baseline_dir"], args["user_model_path"], args["save_dir"], args["save_all"])
