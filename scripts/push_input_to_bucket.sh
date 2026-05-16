#!/bin/bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

echo "create bucket and push"
python scripts/bucket_manager.py \
  --bucket_command "create_bucket" \
  --dir_command "upload_dir"

# python scripts/bucket_manager.py \
#   --bucket_command "delete_bucket"
