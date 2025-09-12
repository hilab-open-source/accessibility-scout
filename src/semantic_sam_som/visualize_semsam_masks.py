from PIL import Image

import numpy as np
import cv2

from .utils.visualizer import Visualizer


def visualize_semsam_masks(image, masks, label_mode="1", alpha=0.1, anno_mode=["Mask"]):
    image_ori = np.asarray(image)

    visual = Visualizer(image_ori)
    label = 1

    mask_map = np.zeros(image_ori.shape, dtype=np.uint8)
    for i, mask in enumerate(masks):
        demo = visual.draw_binary_mask_with_number(
            mask,
            text=str(label),
            label_mode=label_mode,
            alpha=alpha,
            anno_mode=anno_mode,
        )
        # assign the mask to the mask_map
        mask_map[mask == 1] = label
        label += 1
    im = demo.get_image()

    return Image.fromarray(im)

def mark_loc_in_mask(binary_mask):
    binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
    binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
    mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
    mask_dt = mask_dt[1:-1, 1:-1]
    max_dist = np.max(mask_dt)
    coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]

    return coords_x[len(coords_x)//2], coords_y[len(coords_y)//2]
