# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import numpy as np

from semantic_sam.BaseModel import BaseModel
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam import build_model

from .automatic_mask_generator import SemanticSamAutomaticMaskGenerator


def inference_semsam_m2m_auto(model, image, level):
    image_ori = np.asarray(image)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

    mask_generator = SemanticSamAutomaticMaskGenerator(
        model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=10,
        level=level,
    )
    outputs = mask_generator.generate(images)

    sorted_anns = sorted(outputs, key=(lambda x: x["area"]), reverse=True)

    return sorted_anns


def get_semsam_model(cfg_path: str, ckpt_path: str):
    opt_semsam = load_opt_from_config_file(cfg_path)

    model_semsam = (
        BaseModel(opt_semsam, build_model(opt_semsam))
        .from_pretrained(ckpt_path)
        .eval()
        .cuda()
    )

    return model_semsam
