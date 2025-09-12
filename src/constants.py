from typing import Final

# openai parameters
OPENAI_MODEL: Final[str] = "gpt-4o-2024-08-06"
OPENAI_SEED: Final[int] = 42
OPENAI_FREQUENCY_PENALTY: Final[float] = 0.0
OPENAI_TEMPERATURE: Final[float] = 0.8

# prompts
PROMPT_FOLDER: Final[str] = "./env_evaluator/prompts"

# async rate limiting
DELAY_SECONDS = 0
SEMAPHORE_MAX = 100
TIMEOUT = 24

# SOM settings

# SWINL model
SEMSAM_CKPT_PATH = "./ckpts/swinl_only_sam_many2many.pth"
SEMSAM_CFG_PATH = "./configs/semantic_sam_only_sa-1b_swinL.yaml"

# SEMSAM_CKPT_PATH = "./ckpts/swint_only_sam_many2many.pth"
# SEMSAM_CFG_PATh = "./configs/semantic_sam_only_sa-1b_swinT.yaml"

SOM_LEVEL = [3]
SOM_LABEL_MODE = "1"
SOM_ALPHA = 0.1
SOM_ANNO_MODE = ["Mark"] # Mask, Mark, Box
