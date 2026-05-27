cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

echo "distill student"
python app/train_student.py \
  --split "train"

echo "run student_distilled transcripts_filler"
python app/fill_transcripts.py \
  --role "student_distilled" \
  --split "test"

echo "push data to hf"
python app/manage_buckets.py \
  upload-dir "data"
