from pydantic import BaseModel


class Capability(BaseModel):
    name: str
    desc: str
    frequent: bool
    affected_part: str


class UserCapabilities(BaseModel):
    capabilities: list[Capability]


class Task(BaseModel):
    name: str
    desc: str


class EnvTasks(BaseModel):
    tasks: list[Task]


class TaskLocation(BaseModel):
    location: str
    reason: str
    primitives: list[str]


class TaskLocations(BaseModel):
    tasks: list[TaskLocation]


class EnvConcern(BaseModel):
    name: str
    desc: str
    locations: list[int]

class ManuallyAddedEnvConcern(EnvConcern):
    affected_tasks: list[str]


class EnvConcerns(BaseModel):
    concerns: list[EnvConcern]
