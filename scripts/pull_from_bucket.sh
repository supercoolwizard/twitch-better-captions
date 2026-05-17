#!/bin/bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

mkdir -p input
mkdir -p data
mkdir -p models

python scripts/bucket_manager.py \
  "download_dir" "input"

python scripts/bucket_manager.py \
  "download_dir" "data"
