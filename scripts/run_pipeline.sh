#!/bin/bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

echo "folder structure creation"
python app/manage_data.py \
  --audio-ext ".mp3" \
  --text-ext ".txt"

echo "run student transcripts_filler"
python app/transcripts_filler.py \
  --role "student" \
  --split "train"

python app/transcripts_filler.py \
  --role "student" \
  --split "test"

echo "run teacher transcripts_filler"
python app/transcripts_filler.py \
  --role "teacher" \
  --split "train"

python app/transcripts_filler.py \
  --role "teacher" \
  --split "test"

echo "distill student"
python app/train_student.py \
  --split "train"

echo "run student_distilled transcripts_filler"
python app/transcripts_filler.py \
  --role "student_distilled" \
  --split "test"

echo "push data to hf"
python app/manage_buckets.py \
  upload-dir "data"
