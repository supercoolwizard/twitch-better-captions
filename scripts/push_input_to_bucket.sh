#!/bin/bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

echo "folder structure creation"
python app/manage_data.py \
  --audio-ext ".mp3" \
  --text-ext ".txt"

echo "create bucket and push"
python app/manage_buckets.py \
  "create_bucket"

python app/manage_buckets.py \
  upload_dir "input"
