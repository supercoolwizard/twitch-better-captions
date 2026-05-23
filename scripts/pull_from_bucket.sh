#!/bin/bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

python app/manage_buckets.py \
  download-dir "input"

python app/manage_buckets.py \
  download_dir "data"
