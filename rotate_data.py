import os
import json
from PIL import Image, ImageOps


# Load the rotation info
with open('data/image_rotations/image_rotations_HE.json', 'r') as f:
    rotation_info = json.load(f)

input_dir = '../tissue_alignment/data/images/HE_crops_masked'
output_dir = 'data/HE_images_rotated'
os.makedirs(output_dir, exist_ok=True)

for filename, value in rotation_info.items():
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Skip images if marked as "skipped"
    if isinstance(value, dict) and "skipped" in value:
        print(f"Skipping {filename} due to 'skipped' flag.")
        continue

    try:
        img = Image.open(input_path).convert("RGBA")
        rotated = img.rotate(-value, expand=True)

        w, h = rotated.size
        max_dim = max(w, h)

        # Create white square canvas
        square_img = Image.new("RGBA", (max_dim, max_dim), (255, 255, 255, 255))
        x_offset = (max_dim - w) // 2
        y_offset = (max_dim - h) // 2
        square_img.paste(rotated, (x_offset, y_offset), rotated)

        # Convert to RGB (remove alpha) and save
        final_img = square_img.convert("RGB")
        final_img.save(output_path)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue
