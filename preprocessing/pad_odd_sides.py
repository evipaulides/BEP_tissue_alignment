import os
from PIL import Image, ImageOps
from pathlib import Path

def adjust_image_to_odd_16_multiple(image_path, pad_color=(245, 245, 245)):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    def next_odd_multiple_of_16(x):
        return x if (x // 16) % 2 == 1 else x + 16

    # Determine if adjustment is needed
    target_width = next_odd_multiple_of_16(width)
    target_height = next_odd_multiple_of_16(height)

    # Only pad if necessary
    if target_width != width or target_height != height:
        pad_width_total = target_width - width
        pad_height_total = target_height - height
        pad_left = pad_width_total // 2
        pad_right = pad_width_total - pad_left
        pad_top = pad_height_total // 2
        pad_bottom = pad_height_total - pad_top

        padded_image = ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=pad_color
        )
        padded_image.save(image_path)
        print(f"Updated: {image_path.name} → ({target_width}, {target_height})")
    else:
        print(f"Skipped (already OK): {image_path.name}")

def process_image_folder_in_place(folder_path):
    image_paths = list(Path(folder_path).glob("*.png"))

    for img_path in image_paths:
        adjust_image_to_odd_16_multiple(img_path)

# Set your folder path here
folder = "data/HE_images_matched"
process_image_folder_in_place(folder)
