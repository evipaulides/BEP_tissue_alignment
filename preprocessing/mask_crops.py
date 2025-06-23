import os
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def replace_background(images_path, masks_path, output_path, bg_color=(245, 245, 245)):
    """
    Replace the background of images with a specified color using corresponding masks and save the results.

    Parameters:
    images_path (str or Path): Path to the directory containing images.
    masks_path (str or Path): Path to the directory containing masks.
    output_path (str or Path): Path to save the output images with replaced backgrounds.
    bg_color (tuple): RGB color to use as the background (default is light gray).
    """
    images_path = Path(images_path)
    masks_path = Path(masks_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in images_path.iterdir():
        if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        # Load image and corresponding mask
        mask_file = masks_path / img_file.name
        if not mask_file.exists():
            print(f"Mask not found for: {img_file.name}")
            continue

        image = Image.open(img_file).convert("RGB")
        mask = Image.open(mask_file).convert("L")  # Grayscale

        # Create RGB background image
        background = Image.new("RGB", image.size, bg_color)

        # Composite: keep original image where mask > 0, else use background
        image_with_bg = Image.composite(image, background, mask)

        # Save result
        out_path = output_path / img_file.name
        image_with_bg.save(out_path)
        # print(f"Saved: {out_path}")


if __name__ == "__main__":
    # configure paths
    images_path = config.ihc_images_raw  # Path to the directory containing images
    masks_path = config.ihc_masks_raw  # Path to the directory containing masks
    output_path = config.ihc_images_masked  # Path to save the output images

    replace_background(images_path, masks_path, output_path)
