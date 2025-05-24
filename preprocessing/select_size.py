import csv
from PIL import Image
import os

# === Configuration ===
input_csv_path = "data/data_split/val_matches.csv"       # Input CSV file with HE and IHC columns
output_csv_path = "data/data_split/val_filtered.csv"   # Output CSV file for valid matches
he_dir = "data/HE_images_matched"
ihc_dir = "data/IHC_images_matched"
max_area = 1136 * 1232               # Area threshold (change as needed)

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

            if he_area < max_area and ihc_area < max_area:
                filtered_pairs.append({"HE": row["HE"], "IHC": row["IHC"]})
        except Exception as e:
            print(f"Skipping pair ({row['HE']}, {row['IHC']}): {e}")

# === Save Output ===
with open(output_csv_path, mode="w", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["HE", "IHC"])
    writer.writeheader()
    writer.writerows(filtered_pairs)

print(f"Filtered CSV saved to: {output_csv_path}")
