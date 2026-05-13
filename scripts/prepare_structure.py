import argparse

from src.utils.data_manager import DataManager
from src.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="prepare structure (split+transcripts+metadata)"
    )

    parser.add_argument(
        "--audio_ext",
        default=".mp3",
    )
    parser.add_argument(
        "--text_ext",
        default=".txt",
    )
    parser.add_argument(
        "--data_dir",
        default=settings.DATA_DIR,
    )
    parser.add_argument(
        "--input_dir",
        default=settings.INPUT_DIR,
    )

    args = parser.parse_args()

    dm = DataManager(
            args.audio_ext,
            args.text_ext,
            args.data_dir,
            args.input_dir
        )

    dm.train_test_splitter()
    dm.transcripts_files_creator()
    dm.metadata_maker()

if __name__ == "__main__":
    main()
