import base64
from io import BytesIO

from PIL import Image, ImageFile
from pathlib import Path


# Function to encode the image as b64
def pil_to_b64(im: ImageFile) -> str:
    buff = BytesIO()
    im.save(buff, format="jpeg")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return img_str


def openai_img_encode(img_path: Path | str, high_fidelity: bool = True) -> str:
    im = Image.open(img_path)
    im = openai_scale(im, high_fidelity=high_fidelity)

    return pil_to_b64(im)


def openai_scale(im: ImageFile, high_fidelity: bool = True) -> ImageFile:
    # https://platform.openai.com/docs/guides/vision
    w, h = im.size

    if not high_fidelity:  # low fidelity is just a 512x512 image
        im.thumbnail((512, 512))
        return im

    if w > 2048 or h > 2048:  # initial rescale to 2048 x2048 to fit in the square
        im.thumbnail((2048, 2048))

    # scale so that shortest size is 768 pixels
    # do it like this to ensure the shortest side is always exactly 768
    if w < h:
        new_size = (768, int(h * (768 / w)))
    else:
        new_size = (int(w * (768 / h)), 768)

    new_im = im.resize(new_size)

    return new_im
