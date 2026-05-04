from pydantic import DirectoryPath
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    DATA_DIR: DirectoryPath = Path("data")
    INPUT_DIR: DirectoryPath = Path("input")

settings = Settings()
