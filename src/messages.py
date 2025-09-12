import copy
from pathlib import Path

from src.vision import openai_img_encode
import src.constants as const


def get_prompt(name: str, api_type: str = "openai"):
    # prompt_folder = Path(__file__).parent.resolve() / const.PROMPT_FOLDER
    prompt_folder = (
        Path(const.PROMPT_FOLDER) / api_type
    )  # access prompts for a specific api

    with open(prompt_folder / f"{name}.md", "r") as f:
        prompt: str = f.read()

    return prompt


class Messages:
    def __init__(self, load_data: list[dict] = None, api_type="openai"):
        self.api_type = api_type

        if load_data is not None:
            self.msgs = load_data
        else:
            self.msgs = []

    def add_msg(self, role: str, msg: str | list) -> None:
        self.msgs.append({"role": role, "content": msg})

    def add_b64_img_msg(
        self, role: str, msg: str, b64_imgs: list[str], high_fidelity=True
    ) -> None:
        # formats a message with images as well
        content = [{"type": "text", "text": msg}]  # enforces numbered image labels

        fidelity = "high" if high_fidelity else "low"

        for b64_img in b64_imgs:
            payload = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": fidelity,
                },
            }
            content.append(payload)

        self.add_msg(role=role, msg=content)

    def add_oai_img_msg(
        self, role: str, msg: str, img_paths: list[str | Path], high_fidelity=True
    ) -> None:
        # formats a message with images as well

        b64_imgs = [
            openai_img_encode(img_path, high_fidelity=high_fidelity)
            for img_path in img_paths
        ]

        self.add_b64_img_msg(role, msg, b64_imgs, high_fidelity=high_fidelity)

    def to_list(self) -> list[dict]:
        return copy.deepcopy(self.msgs)

    def from_list(self, msg: list[dict]):
        self.msgs = msg

    def clear(self):
        self.msgs = []

    def pop(self, idx: int):
        return self.msgs.pop(idx)

    def set(self, msgs):
        self.msgs = msgs

    def copy(self):
        x = Messages()
        x.set(copy.deepcopy(self.msgs))

        return x

    def __getitem__(self, idx: int):
        return self.msgs[idx]

    def __repr__(self):
        return f"Messages({repr(self.msgs)})"

    def __str__(self):
        return f"Messages({str(self.msgs)})"
