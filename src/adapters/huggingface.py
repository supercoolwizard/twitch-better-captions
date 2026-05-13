from dotenv import load_dotenv
from huggingface_hub import HfApi
import os


def get_hf_username() -> str:
    """
    Loads HF credentials and returns current username.
    Centralized HuggingFace authentication.
    """
    load_dotenv("src/local_settings.env")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not found in env")

    return HfApi().whoami()["name"]


def get_repo_name() -> str:
    username = get_hf_username()
    return f"{username}/moonshine-lora-finetuned"
