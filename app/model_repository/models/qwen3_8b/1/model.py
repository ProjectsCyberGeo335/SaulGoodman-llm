import torch
from loguru import logger
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"DEVICE ACCESS: {DEVICE}")

local_dir = '/home/retro0/cyberspace/projects/ml/llm/saulGoodman-llm/app/llm'
repo_id = 'Qwen/Qwen3-8B'

local_model_path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir
)
