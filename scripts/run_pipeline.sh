#!/bin/bash
set -e

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
