from pathlib import Path
import json
from dataclasses import dataclass, field, asdict
from openai import OpenAI, AsyncOpenAI
from src.sbert import cluster_similar_sentences

import src.constants as const
from src.messages import Messages, get_prompt
import src.helper as h
from src.async_tasks import async_rate_limit_parallelize
import src.response_formats as rf
from src.environment import Environment
from src.vision import pil_to_b64

# for async rate limiting
# something more robust is probably necessary but im lazy


@dataclass
class Task:
    img_name: str = ""
    name: str = ""
    desc: str = ""
    locs: list[str] = field(default_factory=lambda _: [])
    primitives: list[str] = field(default_factory=lambda _: [])


@dataclass
class EnvConcern:
    img_name: str = ""
    name: str = ""
    desc: str = ""
    locs: list[int] = field(default_factory=lambda _: [])
    task_idxs: list[int] = field(default_factory=lambda _: [])
    is_concern: bool = True  # whether its still a concern
    approved: bool = False  # whether the user validated
    manually_entered: bool = False  # whether ai generated
    user_rating: str = ""  # user final comments/ratings


def env_concern_asdict_factory(obj: list[tuple]) -> dict:
    d = {}
    for k, v in obj:
        d[k] = v

    return d


# actual user interface and information management
# per "environment"
class EnvironmentEvaluator:
    def __init__(
        self,
        api_key: str,
        env: Environment | None = None,
        user_model: list[dict] | None = None,
        memory_dir: str | Path | None = None,
        high_fidelity=True,
        log_path: str | None = None,
        init_concerns: bool = True,  # false then just init tasks
    ):
        # llm clients
        self.sync_client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.api_key = api_key

        self.log_path = (
            None if log_path is None else Path(log_path)
        )  # inverted cause python evaluates sequentially

        if (
            env is None or user_model is None
        ) and memory_dir is None:  # not running from memory
            raise ValueError(
                "No environment or user capabilities were passed in with memory not initiated"
            )

        self.env = env
        self.user_model = user_model
        self.high_fidelity = high_fidelity  # for OAI imgs
        self.env_tasks = []
        self.env_concerns = []

        # load from memory if needed
        if memory_dir:
            memory_dir = Path(memory_dir)
            self.load_memory(memory_dir)
        else:
            self.init_env_tasks()
            if init_concerns:
                self.init_env_concerns()

    def refresh_client(self):
        # https://github.com/openai/openai-python/issues/1059
        self.sync_client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    @h.timeit
    def init_env_tasks(self):
        # taking a task based approach where we identify what is most likely done in the environment
        # where might the task be performed in the environment
        # what primitives would each task involve given each environment
        high_level_tasks = self.query_high_level_tasks()
        tasks = self.query_task_locations(high_level_tasks)
        self.env_tasks = self.process_tasks(tasks)

    @h.timeit
    def init_env_concerns(self):
        env_concerns = self.assess_env_concerns()
        # print(json.dumps([asdict(c, dict_factory=env_concern_asdict_factory) for c in env_concerns], indent=4))
        self.env_concerns = self.process_env_concerns(env_concerns)

    @h.timeit
    def query_high_level_tasks(self) -> list[dict]:
        # get the big idea tasks
        self.refresh_client()

        query_tasks_msg = Messages()
        prompt = get_prompt("system-query-all-env-tasks")
        query_tasks_msg.add_msg("system", prompt)

        env_desc = self.env.get_env_desc()
        env_imgs = self.env.get_env_imgs()

        imgs_b64 = [pil_to_b64(x.img) for x in env_imgs]
        query_tasks_msg.add_b64_img_msg(
            "user", env_desc, imgs_b64, high_fidelity=False
        )

        completion = self.sync_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=2048,  # increased max tokens to catch as many as possible
            response_format=rf.EnvTasks,
            messages=query_tasks_msg.to_list(),
        )

        # TODO: CONSIDER HARDCODING SOME IMPORTANT BASIC TASKS
        return json.loads(completion.choices[0].message.parsed.model_dump_json())[
            "tasks"
        ]

    @h.timeit
    def query_task_locations(
        self, high_level_tasks: list[dict]
    ):  # rename to query subtasks
        # gets low level tasks based on the location in the environment to perform a high level task

        env_desc = self.env.get_env_desc()
        env_imgs = self.env.get_env_imgs()

        get_location_tasks = []
        for env_img in env_imgs:
            name, img_b64 = env_img.name, pil_to_b64(env_img.img)
            base_msg = Messages()
            prompt = get_prompt("system-query-task-location")
            base_msg.add_msg("system", prompt)
            base_msg.add_b64_img_msg("user", env_desc, [img_b64], high_fidelity=False)

            for task in high_level_tasks:  # add into the queue
                # have to pass information individually
                task_info = {
                    "img_name": name,
                    "high_level_task": task["name"],
                    "high_level_task_desc": task["desc"],
                }

                get_location_tasks.append(
                    self._query_task_location(task_info, base_msg.copy())
                )

        resp = async_rate_limit_parallelize(get_location_tasks)

        low_level_tasks = []
        for d in resp:
            low_level_tasks.extend(d)

        return low_level_tasks

    async def _query_task_location(self, task: dict, base_msg: Messages):
        # queries all the locations the task might be performed in
        # uses a base message for concurrency easier processing
        # async to improve queries
        # tag helps identify each since we dont know order very well

        base_msg.add_msg("user", f"the task is: {json.dumps(task)}")

        completion = await self.async_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=4096,
            response_format=rf.TaskLocations,
            messages=base_msg.to_list(),
        )

        data = json.loads(completion.choices[0].message.parsed.model_dump_json())[
            "tasks"
        ]

        final_data = []

        for d in data:
            final_data.append(
                Task(
                    task["img_name"],
                    task["high_level_task"].title().strip(),
                    task["high_level_task_desc"].strip(),
                    [d["location"]],
                    d["primitives"],
                )
            )

        return final_data

    def process_tasks(self, task_locations: list):
        # groups low level locations together into high level tasks to save processing
        # ignores extra information about low level tasks but can potentially be added to later
        self.refresh_client()
        final_task_dict = {}

        for task_location in task_locations:
            key = f"{task_location.name}@@{task_location.img_name}"
            # groups tasks and images
            if key not in final_task_dict:
                final_task_dict[key] = task_location
            else:
                final_task_dict[key].locs.extend(task_location.locs)
                final_task_dict[key].primitives.extend(task_location.primitives)

        final_task_list = list(final_task_dict.values())

        for task in final_task_list:  # remove duplications
            task.primitives = list(set(task.primitives))

        return final_task_list

    @h.timeit
    def assess_env_concerns(self):
        self.refresh_client()

        env_desc = self.env.get_env_desc()
        env_imgs = self.env.get_env_imgs()

        concern_tasks = []
        for env_img in env_imgs:
            img_b64 = pil_to_b64(env_img.masked_img)

            assess_msg_tpl = Messages()
            prompt = get_prompt("system-assess-task-accessibility")
            assess_msg_tpl.add_msg("system", prompt)
            assess_msg_tpl.add_msg(
                "user",
                (
                    "The user's description is as follows:"
                    f"{json.dumps(self.user_model, indent=2)}"
                ),
            )
            assess_msg_tpl.add_b64_img_msg("user", env_desc, [img_b64], high_fidelity=self.high_fidelity)

            for task_idx, env_task in enumerate(self.env_tasks):
                if env_task.img_name != env_img.name:  # do by each image
                    continue

                base_msg = assess_msg_tpl.copy()
                high_level_task_prompt = (
                    f"The user is trying to perform {env_task.name.lower()}"
                    f"{env_task.desc.lower()}",
                    f"They could be working at any of the following locations: {env_task.locs}",
                    f"They might be performing any of the following movements while doing so: {env_task.primitives}",
                )
                base_msg.add_msg("user", high_level_task_prompt)
                concern_tasks.append(
                    self._assess_env_concern(task_idx, base_msg.copy())
                )

        concerns = async_rate_limit_parallelize(concern_tasks)

        final_concerns = []
        for concern in concerns:
            final_concerns.extend(concern)

        return final_concerns

    async def _assess_env_concern(self, task_idx: int, base_msg: Messages):
        completion = await self.async_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=4096,
            response_format=rf.EnvConcerns,
            messages=base_msg.to_list(),
        )

        env_concerns = json.loads(
            completion.choices[0].message.parsed.model_dump_json()
        )["concerns"]

        base_msg.add_msg("assistant", json.dumps(env_concerns, indent=2))
        prompt = get_prompt("user-double-check-concerns")
        base_msg.add_msg("user", prompt)

        completion = await self.async_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=4096,
            response_format=rf.EnvConcerns,
            messages=base_msg.to_list(),
        )

        final_data = []
        for concern in env_concerns:
            final_data.append(
                EnvConcern(
                    self.env_tasks[task_idx].img_name,
                    concern["name"].title().strip(),
                    concern["desc"].strip(),
                    concern["locations"],
                    [task_idx],
                )
            )

        return final_data

    def process_env_concerns(self, concerns: list[EnvConcern]):
        # takes a list of concerns
        # removes duplicates

        final_concerns = []

        concern_sentences = [f"{c.name}, {c.desc}" for c in concerns]
        # group similar sentences
        clusters = cluster_similar_sentences(concern_sentences, cosine_threshold=0.75)
        # turns all concerns with similar wording into the same wording
        # combine all duplicate concerns in each image
        env_imgs = self.env.get_env_imgs()

        for cluster in clusters:
            dupe_dict = {img.name: None for img in env_imgs}
            base_concern = concerns[cluster[0]]
            for idx in cluster:
                curr_concern = concerns[idx]

                curr_concern.name = base_concern.name
                curr_concern.desc = base_concern.desc

                # combine the same concerns together for the same image
                curr_img_name = curr_concern.img_name
                if dupe_dict[curr_img_name] is None:
                    dupe_dict[curr_img_name] = curr_concern
                else:
                    dupe_dict[curr_img_name].locs.extend(curr_concern.locs)
                    dupe_dict[curr_img_name].task_idxs.extend(curr_concern.task_idxs)

            for key, val in dupe_dict.items():
                if val is None:
                    continue
                dupe_dict[key].locs = list(set(dupe_dict[key].locs))
                dupe_dict[key].task_idxs = list(set(dupe_dict[key].task_idxs))
                final_concerns.append(dupe_dict[key])

        return final_concerns

    def approve_concern(self, concern_idx: str, is_concern: bool, rating: str):
        # concern id to get concern
        # is_concern on whether its a concern or not
        # rating for user entered reasoning on why or why not
        # update will rerun the concern with the new information
        concern = self.env_concerns[concern_idx]
        concern.is_concern = is_concern
        concern.approved = True
        concern.user_rating = rating

    def update_user_model(self, new_user_model: dict[list]):
        self.user_model = new_user_model
        self.init_env_concerns()  # re init the concerns with the new user model

    @h.timeit
    def add_concern(self, img_name: str, concern_msg: str):
        # creates a concern from a general user description
        self.refresh_client()

        add_concern_msg = Messages()
        prompt = get_prompt("system-add-concern-from-msg")
        add_concern_msg.add_msg("system", prompt)
        add_concern_msg.add_msg("user", f"My capabilities are:\n {self.user_model}")
        add_concern_msg.add_msg(
            "user",
            f"Some tasks I might perform are:\n {json.dumps(self.get_env_tasks(as_dict=True), indent=2)}",
        )

        env_desc = self.env.get_env_desc()
        env_imgs = self.env.get_env_imgs()

        img_names = [img.name for img in env_imgs]
        if img_name not in img_names:
            print("image not in env.")
            return None

        env_img = env_imgs[img_names.index(img_name)]
        imgs_b64 = [pil_to_b64(env_img.masked_img)]
        add_concern_msg.add_b64_img_msg(
            "user", env_desc, imgs_b64, high_fidelity=self.high_fidelity
        )

        add_concern_msg.add_msg(
            "user", f"I see the following concern to identify: {concern_msg}"
        )

        completion = self.sync_client.beta.chat.completions.parse(
            model=const.OPENAI_MODEL,
            seed=const.OPENAI_SEED,
            frequency_penalty=const.OPENAI_FREQUENCY_PENALTY,
            temperature=const.OPENAI_TEMPERATURE,
            max_tokens=2048,  # increased max tokens to catch as many as possible
            response_format=rf.ManuallyAddedEnvConcern,
            messages=add_concern_msg.to_list(),
        )

        # TODO: CONSIDER HARDCODING SOME IMPORTANT BASIC TASKS
        added_concern = json.loads(
            completion.choices[0].message.parsed.model_dump_json()
        )
        task_idx_list = []

        # find the task idx
        task_names = [task.name.lower() for task in self.env_tasks]
        for task in added_concern["affected_tasks"]:
            try:
                task_idx = task_names.index(task.strip().lower())
                task_idx_list.append(task_idx)
            except ValueError:
                print(f"Task does not exist: {task}")
                continue

        self.env_concerns.append(
            EnvConcern(
                img_name,
                added_concern["name"],
                added_concern["desc"],
                added_concern["locations"],
                task_idx_list,
                manually_entered=True,
            )
        )

        return self.env_concerns[-1]

    def get_env_tasks(self, as_dict=False):
        if as_dict:
            return [asdict(task) for task in self.env_tasks]
        else:
            return self.env_tasks

    def get_env_concerns(self, as_dict=False):
        if as_dict:
            return [
                asdict(c, dict_factory=env_concern_asdict_factory)
                for c in self.env_concerns
            ]
        else:
            return self.env_concerns

    def save_memory(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "env_tasks.json", "w") as f:
            json.dump(self.get_env_tasks(as_dict=True), f, indent=4)

        out_concerns = self.get_env_concerns(as_dict=True)
        with open(save_dir / "env_concerns.json", "w") as f:
            json.dump(out_concerns, f, indent=4)

        with open(save_dir / "user_capabilities.json", "w") as f:
            json.dump(self.user_model, f, indent=4)

        self.env.save_environment(save_dir / "env")

    def load_memory(self, load_dir: Path):
        with open(load_dir / "env_tasks.json", "r") as f:
            tasks = json.load(f)
        self.env_tasks = []
        for task in tasks:
            self.env_tasks.append(Task(*list(task.values())))

        with open(load_dir / "env_concerns.json", "r") as f:
            env_concerns = json.load(f)
        self.env_concerns = []
        for concern in env_concerns:
            self.env_concerns.append(EnvConcern(*list(concern.values())))

        with open(load_dir / "user_capabilities.json", "r") as f:
            self.user_model = json.load(f)

        self.env = Environment(load_dir / "env")

    def is_all_approved(self) -> bool:
        for concern in self.env_concerns:
            if not concern.approved:
                return False

        return True

    def get_env(self) -> Environment:
        return self.env
