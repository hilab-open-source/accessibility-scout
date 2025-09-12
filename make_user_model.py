
import os
import argparse
from dotenv import load_dotenv
import json

from src.user_modeler import UserModeler

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Environment Accessibility Evaluation Environment Cache Maker",
        description="Creates a user model from text",
    )

    parser.add_argument(
        "--user-model-text",
        type=str,
        help="Textual description of user capabilities",
        required=False,
        default="",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        help="Directory to save all the final baselines",
        required=True,
    )

    return parser


def main(user_model_text: str, save_path: str):
    modeler = UserModeler(api_key=API_KEY, initial_user_desc=user_model_text)

    user_model = modeler.get_user_model(as_dict=True)

    with open(save_path, "w") as f:
        json.dump(user_model, f, indent=4)

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    main(args["user_model_text"], args["save_path"])
