import os
import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

""" This script selects matched HE and IHC images based on a JSON file containing matches."""

# ==== CONFIG ====
json_path = config.matches_file  # Path to your JSON file
source_he_folder = config.he_images_rotated  # Folder containing original HE images
source_ihc_folder = config.ihc_images_rotated  # Folder containing original IHC images
dest_he_folder = config.he_dir  # Destination folder for matched HE images
dest_ihc_folder = config.ihc_dir  # Destination folder for matched IHC images
# ================

# Make sure destination folders exist
os.makedirs(dest_he_folder, exist_ok=True)
os.makedirs(dest_ihc_folder, exist_ok=True)

# Load the JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Iterate through the data
for case_id, matches in data.items():
    if isinstance(matches, str) and matches == "NO_MATCHES":
        print(f"Skipping case {case_id}: no matches.")
        continue

    for match_id, stains in matches.items():
        he_filename = stains.get("he")
        ihc_filename = stains.get("ihc")

        if he_filename:
            src_he = os.path.join(source_he_folder, he_filename)
            dst_he = os.path.join(dest_he_folder, he_filename)
            if os.path.exists(src_he):
                shutil.copy2(src_he, dst_he)
            else:
                print(f"WARNING: HE image not found: {src_he}")

        if ihc_filename:
            src_ihc = os.path.join(source_ihc_folder, ihc_filename)
            dst_ihc = os.path.join(dest_ihc_folder, ihc_filename)
            if os.path.exists(src_ihc):
                shutil.copy2(src_ihc, dst_ihc)
            else:
                print(f"WARNING: IHC image not found: {src_ihc}")
