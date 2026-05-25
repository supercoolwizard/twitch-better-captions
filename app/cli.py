import typer

from app.manage_buckets import app as manage_buckets
from app.manage_data import app as fill_transcripts
from app.train_student import app as manage_data
from app.fill_transcripts import app as train_student

app = typer.Typer()

app.add_typer(manage_buckets, name="manage_buckets")
app.add_typer(fill_transcripts, name="fill_transcripts")
app.add_typer(manage_data, name="manage_data")
app.add_typer(train_student, name="train_student")

if __name__ == "__main__":
    app()
