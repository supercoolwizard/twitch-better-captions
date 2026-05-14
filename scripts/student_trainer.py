from src.fine_tuning.student_trainer import StudentTrainer
from src.adapters.huggingface import get_hf_token, get_repo_name
from src.config import settings
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="train student_model"
    )

    parser.add_argument()

    token = get_hf_token()
    repo = get_repo_name()

    student_trainer = StudentTrainer(settings, token, repo)
    processed_dataset = student_trainer.load_and_preprocess_data()

    student_trainer.train(processed_dataset)


if __name__ == "__main__":
    main()
