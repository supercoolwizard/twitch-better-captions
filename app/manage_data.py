import typer
from pathlib import Path
from src.services.data_manager import DataManager
from src.config import settings

app = typer.Typer()


@app.command()
def main(
    audio_ext: str = typer.Option(
        ".mp3",
        help="Audio file extension",
    ),
    text_ext: str = typer.Option(
        ".txt",
        help="Transcript file extension",
    ),
    data_dir: str = typer.Option(
        settings.DATA_DIR,
        help="Data directory",
    ),
    input_dir: str = typer.Option(
        settings.INPUT_DIR,
        help="Input directory",
    ),
):
    dm = DataManager(
        audio_ext,
        text_ext,
        Path(data_dir),
        Path(input_dir),
    )

    dm.train_test_splitter()
    dm.transcripts_files_creator()
    dm.metadata_maker()


if __name__ == "__main__":
    app()
