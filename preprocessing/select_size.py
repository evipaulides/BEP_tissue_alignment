import csv
from PIL import Image
import os
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

""" This script filters HE and IHC image pairs based on their area, ensuring they are below a specified threshold.
It reads a CSV file containing pairs of HE and IHC images, checks their dimensions, and writes valid pairs to a new CSV file.
"""

# === Configuration ===
input_csv_path = config.train_csv       # Input CSV file with HE and IHC columns
output_csv_path = "data/data_split/train_filtered.csv"   # Output CSV file for valid matches
# ^ change to config.val_csv or config.test_csv for validation or test sets
# ^ change csv path in config.py to filtered csv after running this script

he_dir = config.he_dir          # Directory containing HE images
ihc_dir = config.ihc_dir          # Directory containing IHC images
max_area = config.max_image_size         # Area threshold (change as needed)

apply_aug = False  # Set to True if you want to apply augmentation area calculations (in case of training data)
# === Process ===
filtered_pairs = []

with open(input_csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        he_path = os.path.join(he_dir, row["HE"])
        ihc_path = os.path.join(ihc_dir, row["IHC"])

        try:
            he_img = Image.open(he_path)
            ihc_img = Image.open(ihc_path)

            he_area = he_img.width * he_img.height
            ihc_area = ihc_img.width * ihc_img.height
            if apply_aug:
                augmented_he_area = (math.cos(5/ 180 * math.pi) * he_img.width + math.sin(5 / 180 * math.pi) * he_img.height) * (math.cos(5 / 180 * math.pi) * he_img.height + math.sin(5 / 180 * math.pi) * he_img.width)
                augmented_ihc_area = (math.cos(5 / 180 * math.pi) * ihc_img.width + math.sin(5 / 180 * math.pi) * ihc_img.height) * (math.cos(5 / 180 * math.pi) * ihc_img.height + math.sin(5 / 180 * math.pi) * ihc_img.width)
            else:
                augmented_he_area = 0
                augmented_ihc_area = 0

            if he_area < max_area and ihc_area < max_area and augmented_he_area < max_area and augmented_ihc_area < max_area:
                filtered_pairs.append({"HE": row["HE"], "IHC": row["IHC"]})
        except Exception as e:
            print(f"Skipping pair ({row['HE']}, {row['IHC']}): {e}")

# === Save Output ===
with open(output_csv_path, mode="w", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["HE", "IHC"])
    writer.writeheader()
    writer.writerows(filtered_pairs)

print(f"Filtered CSV saved to: {output_csv_path}")
