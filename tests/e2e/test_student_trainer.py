from src.config import settings
from src.fine_tuning.student_trainer import StudentTrainer
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv("src/local_settings.env")
hf_token = os.getenv("HF_TOKEN")
hf_username = HfApi().whoami()["name"]
hf_model_repo = f"{hf_username}/moonshine-lora-finetuned"


student_trainer = StudentTrainer(settings, hf_token, hf_model_repo)
processed_dataset = student_trainer.load_and_preprocess_data()

student_trainer.train(processed_dataset)
