#!/bin/bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

echo "folder structure creation"
python scripts/prepare_structure.py

echo "pull the input bucket from hf"
python scripts/bucket_manager.py \
  --dir_command "download_dir"

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

