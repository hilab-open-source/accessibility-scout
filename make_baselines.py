
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

MAX_CONCURRENT = 2


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
        description="Creates environment data structure + masks for LLM input",
    )

    parser.add_argument(
        "--env-dir",
        type=str,
        help="Directory of configured environments. Each folder should be an individual environment to process",
        required=True,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save all the final baselines",
        required=True,
    )

    parser.add_argument(
        "--user-model-text",
        type=str,
        help="Textual description of user capabilities",
        required=False,
        default="",
    )

    return parser


def make_baseline_task(args):

    env_path, save_path, user_model = args
    try:
        with HiddenPrints(): # hides all the printing
            env = Environment(env_path)
            evaluator = EnvironmentEvaluator(
                api_key=API_KEY, env=env, user_model=user_model, init_concerns=False
            )
            evaluator.save_memory(save_path)
    except Exception as e:
        print(e)
        return False

    return True


def main(env_dir: str, user_model_text: str, save_dir: str):
    env_dir = Path(env_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    modeler = UserModeler(api_key=API_KEY, initial_user_desc=user_model_text)

    user_model = modeler.get_user_model(as_dict=True)

    with open(save_dir / "user_model.json", "w") as f:
        json.dump(user_model, f, indent=4)

    args = [(d, save_dir / d.stem, user_model) for d in env_dir.iterdir() if d.is_dir()]

    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        results = list(tqdm(executor.map(make_baseline_task, args), total=len(args)))

    # make_baseline_task(args[0])


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    main(args["env_dir"], args["user_model_text"], args["save_dir"])
