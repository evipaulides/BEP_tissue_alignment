import json
import os
import shutil
from pathlib import Path

def collect_images_with_empty_values(json_path, source_folder, dest_folder):
    # Ensure destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Loop over each key and check if it has an empty value
    for key, value in data.items():
        if value in (None, "", {}, []):
            # Search for all matching files in the source folder
            for file_name in os.listdir(source_folder):
                if file_name.startswith(f"{key}_"):
                    src_path = os.path.join(source_folder, file_name)
                    dst_path = os.path.join(dest_folder, file_name)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {file_name}")

collect_images_with_empty_values(
    json_path='image_matches.json',
    source_folder='data/HE_images_rotated',
    dest_folder='data/HE_empty_match'
)
