cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

echo "folder structure creation"
python app/manage_data.py \
  --audio-ext ".mp3" \
  --text-ext ".txt"

echo "run student transcripts_filler"
python app/fill_transcripts.py \
  --role "student" \
  --split "train"

python app/fill_transcripts.py \
  --role "student" \
  --split "test"

echo "push data to hf"
python app/manage_buckets.py \
  upload-dir "data"

echo "run teacher transcripts_filler"
python app/fill_transcripts.py \
  --role "teacher" \
  --split "train"

python app/fill_transcripts.py \
  --role "teacher" \
  --split "test"

echo "push data to hf"
python app/manage_buckets.py \
  upload-dir "data"
