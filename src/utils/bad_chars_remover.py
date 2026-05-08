import os
import pandas as pd
import re

def clean_name(name):
    new_name = name.replace('[', '').replace(']', '').replace(' ', '_')
    return re.sub(r'_+', '_', new_name)

base_dirs = ['data/train', 'data/test']

for base in base_dirs:
    if not os.path.exists(base):
        continue

    print(f"Processing directory: {base}")

    for root, dirs, files in os.walk(base):
        for f in files:
            if f == 'metadata.csv':
                continue

            old_path = os.path.join(root, f)
            new_filename = clean_name(f)
            new_path = os.path.join(root, new_filename)

            if old_path != new_path:
                os.rename(old_path, new_path)

    metadata_path = os.path.join(base, 'metadata.csv')
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path, sep=',')

        cols_to_fix = ['file_name', 'audio_path', 'student_path', 'teacher_path']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = df[col].apply(clean_name)

        df.to_csv(metadata_path, sep=',', index=False)
        print(f"Updated {metadata_path}")

print("Cleanup complete!")
