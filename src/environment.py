from pathlib import Path
import copy
import json
from dataclasses import dataclass, field
from PIL import Image, ImageFile

import numpy as np

import src.constants as const
from src.semantic_sam_som.visualize_semsam_masks import (
    visualize_semsam_masks,
    mark_loc_in_mask,
)


@dataclass
class EnvImg:
    name: str
    img: ImageFile.ImageFile = field(repr=False)
    masks: list[np.ndarray] = field(default_factory=lambda _: [], repr=False)
    masked_img: ImageFile.ImageFile = field(default=None, repr=False)
    mark_locs: list[tuple[int, int]] = field(default_factory=lambda _: [])


def load_mask_dir(mask_dir: Path):
    # loads all the np files in a mask dir
    # returns empty list if doesnt exist
    if not mask_dir.exists():
        return []

    masks = []

    # ensures order
    ind_mask_paths = list(mask_dir.iterdir())
    ind_mask_paths.sort(key=lambda x: int(x.stem))
    for ind_mask_path in ind_mask_paths:
        if ind_mask_path.suffix != ".npy":
            continue
        masks.append(np.load(ind_mask_path))

    return masks


def calc_mask_mark_locations(masks):
    mark_locs = []
    for mask in masks:
        coords = mark_loc_in_mask(mask)
        mark_locs.append(coords)

    return mark_locs


class Environment:  # might need to use this to create a cache later for the image mask
    def __init__(self, load_dir: str | Path | None=None, high_fidelity=True):
        self.env_desc = ""
        self.high_fidelity = high_fidelity

        self.env_imgs: EnvImg = []

        if load_dir is not None:
            load_dir = Path(load_dir)
            self.load_environment(load_dir)

    def save_environment(self, save_dir: str | Path, save_masked_imgs=True, quality=100):
        save_dir = Path(save_dir)

        img_save_dir = save_dir / "imgs"
        mask_save_dir = save_dir / "masks"
        masked_imgs_save_dir = save_dir / "masked_imgs"

        img_save_dir.mkdir(parents=True, exist_ok=True)
        mask_save_dir.mkdir(parents=True, exist_ok=True)
        masked_imgs_save_dir.mkdir(parents=True, exist_ok=True)

        for env_img in self.env_imgs:
            name = env_img.name
            env_img.img.save(img_save_dir / f"{name}.jpg")
            masks = env_img.masks

            if masks != []:
                curr_mask_save_dir = mask_save_dir / name
                curr_mask_save_dir.mkdir(exist_ok=True)

                # save each mask as an individual numpy array
                for i, ind_mask in enumerate(masks):
                    np.save(curr_mask_save_dir / f"{i}.npy", ind_mask)

                if save_masked_imgs:
                    env_img.masked_img.save(masked_imgs_save_dir / f"{name}.jpg", optimize=True, quality=quality)

        metadata = {
            "high_fidelity": self.high_fidelity,
            "env_desc": self.env_desc,
        }

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def load_environment(self, load_dir: str | Path):
        # load metadata
        metadata_path = load_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.high_fidelity = metadata["high_fidelity"]
            self.env_desc = metadata["env_desc"]

        imgs_dir = load_dir / "imgs"
        masks_dir = load_dir / "masks"
        masked_imgs_dir = load_dir / "masked_imgs"

        self.env_imgs = []
        img_paths = list(imgs_dir.iterdir())
        for img_path in img_paths:
            img_name = img_path.stem
            img = Image.open(img_path)

            curr_mask_dir = masks_dir / img_name
            masks = load_mask_dir(curr_mask_dir)

            # loads from environment if it exists otherwise generates it
            masked_img_path = masked_imgs_dir / f"{img_name}.jpg"
            masked_img = None
            if masked_img_path.exists():
                masked_img = Image.open(masked_img_path)

            self.add_img(img_name, img, masks, masked_img)


    def add_img(self, img_name: str, img: ImageFile.ImageFile, masks: None | list[np.ndarray]=None, masked_img: None | ImageFile.ImageFile=None):
        if masks is not None:
            if masked_img is None: # load the masked img if its alr a file
                masked_img = visualize_semsam_masks(  # might want to load this
                    img,
                    masks,
                    const.SOM_LABEL_MODE,
                    const.SOM_ALPHA,
                    const.SOM_ANNO_MODE,
                )
            mark_locs = calc_mask_mark_locations(masks)
        else:
            masks = []
            masked_img = None
            mark_locs = []

        self.env_imgs.append(
            EnvImg(img_name, img, masks, masked_img, mark_locs)
        )

    def get_env_imgs(self):
        return copy.deepcopy(self.env_imgs)

    def set_env_desc(self, env_desc) -> str:
        self.env_desc = env_desc

    def get_env_desc(self) -> str:
        return self.env_desc
