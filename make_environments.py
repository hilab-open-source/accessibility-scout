# separate file to create a bunch of environment data caches
# run separately to allow users to avoid making all the SOM/Semantic-SAM imports

import argparse
from PIL import Image
from pathlib import Path

from tqdm import tqdm

from src.vision import openai_scale
from src.environment import Environment
import src.constants as const
from src.semantic_sam_som.inference_semsam_m2m_auto import (
    inference_semsam_m2m_auto,
    get_semsam_model,
)


def make_environment(
    model, img_path: str | Path, env_desc: str = ""
) -> None:
    img_path = Path(img_path)
    env = Environment()

    env.set_env_desc(env_desc)

    img = openai_scale(Image.open(img_path))

    mask = inference_semsam_m2m_auto(model, img, const.SOM_LEVEL)
    seg_masks = [m["segmentation"] for m in mask]

    env.add_img(img_path.stem, img, seg_masks)

    return env


def main(env_dir: str, save_dir: str):
    env_dir = Path(env_dir)
    save_dir = Path(save_dir)

    save_dir.mkdir(exist_ok=True)

    model = get_semsam_model(const.SEMSAM_CFG_PATH, const.SEMSAM_CKPT_PATH)

    env_paths = list(env_dir.iterdir())

    num = 0
    for i, img_path in tqdm(enumerate(env_paths)):
        if img_path.suffix not in (".png", ".jpg", ".jpeg", ".JPG"):
            continue

        save_path = save_dir / str(num)
        num += 1

        env = make_environment(model, img_path)
        env.save_environment(save_path, quality=95)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Environment Accessibility Evaluation Environment Cache Maker",
        description="Creates environment data structure + masks for LLM input",
    )

    parser.add_argument(
        "--env-dir",
        type=str,
        help="Directory of environment images. Each folder should be an individual image to process",
        required=True,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save all the final environments",
        required=True,
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    main(args["env_dir"], args["save_dir"])
