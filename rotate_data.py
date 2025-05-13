import os
import json
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def rotate_image(img, mask, value):
    # Check if the image is in RGBA mode
    if img.mode != 'RGBA':
        # Convert to RGBA if not already in that mode
        img = img.convert("RGBA")

    # Check if the mask is in RGBA mode
    if mask.mode != 'L':
        # Convert to RGBA if not already in that mode
        mask = mask.convert("L")

    # Rotate the image
    rotated = img.rotate(-value, expand=True, resample=Image.BICUBIC, fillcolor=(245, 245, 245, 255))
    rotated_mask = mask.rotate(-value, expand=True, resample=Image.NEAREST, fillcolor=0)

    # Crop mask to the size of white pixels
    bbox = rotated_mask.getbbox()
    if bbox:
        cropped_mask = rotated_mask.crop(bbox)
    else:
        print("No white pixels found in the mask. Skipping cropping.")
        return img
    
    # Reduce the image to the size of the mask
    img_cropped = rotated.crop(bbox)
    if img_cropped.size != cropped_mask.size:
        print(f"Image and mask sizes do not match after cropping: {img_cropped.size} vs {cropped_mask.size}.")
        return img
    
    # Convert to RGB (remove alpha) and save
    img_cropped = img_cropped.convert("RGB")

    return img_cropped, cropped_mask

def cropped_image_to_input(img, mask):
    #  Add padding until the image is a multiple of 16
    target_size = 16
    width, height = img.size
    new_width = (width + target_size - 1) // target_size * target_size
    new_height = (height + target_size - 1) // target_size * target_size
    padded_img = Image.new("RGB", (new_width, new_height), (245, 245, 245))
    padded_img.paste(img, ((new_width - width) // 2, (new_height - height) // 2))

    # add 0 padding to the mask to make it the same size as the image
    padded_mask = Image.new("L", (new_width, new_height), 0)
    padded_mask.paste(mask, ((new_width - width) // 2, (new_height - height) // 2))
    
    return padded_img, padded_mask

def plot_images(img, mask, img_input, mask_input):
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    axes[0,0].imshow(img)
    axes[0,0].set_title("Image")
    axes[0,0].axis("off")

    axes[0,1].imshow(mask, cmap="gray")
    axes[0,1].set_title("Mask")
    axes[0,1].axis("off")

    axes[1,0].imshow(img_input)
    axes[1,0].set_title("Padded Image")
    axes[1,0].axis("off")

    axes[1,1].imshow(mask_input, cmap="gray")
    axes[1,1].set_title("Padded Mask")
    axes[1,1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Load the rotation info
    # with open('data/image_rotations/image_rotations_IHC_pt4.json', 'r') as f:
    with open('image_rotations_HE_strip.json', 'r') as f:
        rotation_info = json.load(f)

    input_dir = '../tissue_alignment/data/images/HE_crops_masked'
    input_mask_dir = '../tissue_alignment/data/annotations/HE_crops'
    output_dir = 'data/HE_images_rotated'
    output_mask_dir = 'data/HE_masks_rotated'
    os.makedirs(output_dir, exist_ok=True)

    #counter = 0

    for filename, value in rotation_info.items():
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        mask_path = os.path.join(input_mask_dir, filename)

        # Skip images if marked as "skipped"
        if isinstance(value, dict) and "skipped" in value:
            print(f"Skipping {filename} due to 'skipped' flag.")
            continue
        # if filename.split('_')[0] != '2074':
        #     continue

        try:
            img = Image.open(input_path)
            mask = Image.open(mask_path)
            img, mask = rotate_image(img, mask, value)

            img_input, mask_input = cropped_image_to_input(img, mask)
            
            #print the size of the resulting image and mask
            #print(f"Image size: {img_input.size}, Mask size: {mask_input.size}")

            # Save image and mask
            img_input.save(os.path.join(output_dir, filename))
            mask_input.save(os.path.join(output_mask_dir, filename))

            #counter += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        #if counter >= 2:
            #break
