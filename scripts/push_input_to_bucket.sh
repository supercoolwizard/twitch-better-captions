#!/bin/bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

mkdir -p input
mkdir -p data
mkdir -p models
touch -c src/local_settings.env

echo "folder structure creation"
python scripts/prepare_structure.py

echo "create bucket and push"
python scripts/bucket_manager.py \
  "create_bucket"

python scripts/bucket_manager.py \
  "upload_dir" "input"

# python scripts/bucket_manager.py \
#   --bucket_command "delete_bucket"
