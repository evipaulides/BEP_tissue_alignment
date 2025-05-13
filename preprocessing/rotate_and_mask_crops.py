import os
from PIL import Image
from pathlib import Path

def replace_background(images_path, masks_path, output_path, bg_color=(245, 245, 245)):
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
    # VARIABLES
    images_path = "C:/Users/20223399/OneDrive - TU Eindhoven/TUe Biomedical Engineering/Year 3/Q4/BEP_tissue_alignment/tissue_alignment/data/images/IHC_crops"  # Path to the directory containing images
    masks_path = "C:/Users/20223399/OneDrive - TU Eindhoven/TUe Biomedical Engineering/Year 3/Q4/BEP_tissue_alignment/tissue_alignment/data/annotations/IHC_crops"  # Path to the directory containing masks
    output_path = "C:/Users/20223399/OneDrive - TU Eindhoven/TUe Biomedical Engineering/Year 3/Q4/BEP_tissue_alignment/tissue_alignment/data/images/IHC_crops_masked"  # Path to save the output images

    replace_background(images_path, masks_path, output_path)
