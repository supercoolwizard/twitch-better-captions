#!/bin/bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

mkdir -p input
mkdir -p data
touch -c src/local_settings.env

echo "pull the input bucket from hf"
python scripts/bucket_manager.py \
  "download_dir" "input"

echo "folder structure creation"
python scripts/prepare_structure.py

echo "run student transcripts_filler"
python scripts/transcripts_filler.py \
  --model "student"

echo "run teacher transcripts_filler"
python scripts/transcripts_filler.py \
  --model "teacher"

echo "distill student"
python scripts/student_trainer.py

echo "run student_distilled transcripts_filler"
python scripts/transcripts_filler.py \
  --model "student_distilled"

echo "push data to hf"
python scripts/bucket_manager.py \
  --upload_dir "data"

