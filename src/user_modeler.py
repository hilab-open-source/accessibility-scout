import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal

from openai import OpenAI, AsyncOpenAI

import src.constants as const
from src.messages import Messages, get_prompt
import src.helper as h
from src.vision import pil_to_b64
import src.response_formats as rf
from src.environment import Environment


@dataclass
class UserModelAttribute:
    name: str
    desc: str
    frequent: bool
    affected_part: Literal[
        "arms", "legs", "feet", "back", "chest", "hands", "eyes", "ears", "preference"
    ]


class UserModeler:
    def __init__(
        self,
        api_key: str,
        initial_user_desc: str | None = None,
        memory_path: str | Path | None = None,
    ):
        # llm clients
        self.sync_client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.api_key = api_key
        self.initial_user_desc = initial_user_desc

        if memory_path:
            memory_path = Path(memory_path)
            self.load_memory(memory_path)

        elif initial_user_desc is not None:
            self.user_model = self.model_from_text_desc(self.initial_user_desc)
        else:
            print("No memory/initialization loaded for modeling")
            self.user_model = {}

    @h.timeit
    def model_from_text_desc(self, desc: str):
        model_msgs = Messages()
        prompt = get_prompt("system-user-model-from-text")
        model_msgs.add_msg("system", prompt)
        model_msgs.add_msg("user", desc)

        completion = self.sync_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=2048,
            response_format=rf.UserCapabilities,
            messages=model_msgs.to_list(),
        )

        model = json.loads(completion.choices[0].message.parsed.model_dump_json())[
            "capabilities"
        ]

        user_model = []
        for att in model:
            user_model.append(
                UserModelAttribute(
                    att["name"], att["desc"], att["frequent"], att["affected_part"]
                )
            )

        return user_model

    def update_user_model(
        self, update_msg: str
    ):  # TODO: Figure out something better here to update
        update_msgs = Messages()
        prompt = get_prompt("system-user-model-from-text")
        update_msgs.add_msg("system", prompt)
        update_msgs.add_msg("user", update_msg)

        completion = self.sync_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=2048,
            response_format=rf.UserCapabilities,
            messages=update_msgs.to_list(),
        )

        model = json.loads(completion.choices[0].message.parsed.model_dump_json())[
            "capabilities"
        ]

        self.add_user_model(model)

    @h.timeit
    def update_by_env_feedback(self, env: Environment, feedback: str):
        env_imgs = env.get_env_imgs()
        env_desc = env.get_env_desc()
        update_msgs = Messages()
        prompt = get_prompt("system-update-user-model-from-env-feedback")
        update_msgs.add_msg("system", prompt)

        msg = f"Environment Description: {env_desc}, Feedback: {feedback}"
        update_msgs.add_b64_img_msg("user", msg, [pil_to_b64(env_img.img) for env_img in env_imgs])

        completion = self.sync_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=2048,
            response_format=rf.UserCapabilities,
            messages=update_msgs.to_list(),
        )

        model = json.loads(completion.choices[0].message.parsed.model_dump_json())[
            "capabilities"
        ]

        self.add_user_model(model)

    @h.timeit
    def update_by_env_ann(self, env: Environment, env_concerns: list):
        env_imgs = env.get_env_imgs()
        env_desc = env.get_env_desc()
        update_msgs = Messages()
        prompt = get_prompt("system-update-user-model-from-env")
        update_msgs.add_msg("system", prompt)

        for env_img in env_imgs:
            evaluated_concerns = []
            for concern in env_concerns:
                if concern.img_name != env_img.name:
                    continue

                if not concern.approved:
                    continue

                rating_str = ""
                if concern.user_rating == "":
                    rating_str = f"I think {concern.name} about {concern.desc} is {concern.is_concern}"
                else:
                    rating_str = f"I think {concern.name} about {concern.desc} is {concern.is_concern} because {concern.user_rating}"
                evaluated_concerns.append(rating_str)

            msg = f"Environment Description: {env_desc}, Concerns: {json.dumps(evaluated_concerns, indent=2)}"
            update_msgs.add_b64_img_msg("user", msg, [pil_to_b64(env_img.img)])

        completion = self.sync_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=2048,
            response_format=rf.UserCapabilities,
            messages=update_msgs.to_list(),
        )

        model = json.loads(completion.choices[0].message.parsed.model_dump_json())[
            "capabilities"
        ]

        self.add_user_model(model)


    def add_user_model(self, added_user_model: list):
        # could also concatenate similar ones through sentence similarity
        update_msgs = Messages()
        prompt = get_prompt("system-add-user-model")
        update_msgs.add_msg("user", prompt)
        update_msgs.add_msg(
            "user", json.dumps(self.get_user_model(as_dict=True), indent=2)
        )
        update_msgs.add_msg("user", json.dumps(added_user_model, indent=2))

        completion = self.sync_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=2048,
            response_format=rf.UserCapabilities,
            messages=update_msgs.to_list(),
        )

        model = json.loads(completion.choices[0].message.parsed.model_dump_json())[
            "capabilities"
        ]

        user_model = []
        for att in model:
            user_model.append(
                UserModelAttribute(
                    att["name"], att["desc"], att["frequent"], att["affected_part"]
                )
            )

        self.user_model = user_model

    def get_user_model(self, as_dict=False):
        if as_dict:
            return [asdict(att) for att in self.user_model]
        else:
            # TODO: DEEPCOPY ALL GET MESSAGES
            return self.user_model

    def load_memory(self, load_file: Path | str):
        with open(load_file, "r") as f:
            user_model = json.load(f)

            self.user_model = []

            for att in user_model:
                self.user_model.append(
                    UserModelAttribute(
                        att["name"], att["desc"], att["frequent"], att["affected_part"]
                    )
                )

    def save_memory(self, save_path: Path | str):
        with open(save_path, "w") as f:
            json.dump(self.get_user_model(as_dict=True), f, indent=4)
