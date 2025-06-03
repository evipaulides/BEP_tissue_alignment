import os
import json
from PIL import Image
import numpy as np
import math


def crop_to_mask(image, mask):
    """ Crop the image and mask to the bounding box of the mask.
    
    Parameters:
    image (PIL.Image): The image to be cropped.
    mask (PIL.Image): The mask to be used for cropping.
    
    Returns:
    cropped_image (PIL.Image): The cropped image.
    cropped_mask (PIL.Image): The cropped mask. """

    # Calculate the bounding box of the mask
    np_mask = np.array(mask)
    y_indices, x_indices = np.where(np_mask > 0)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return image, mask  # Avoid cropping everything
    left, right = x_indices.min(), x_indices.max()
    top, bottom = y_indices.min(), y_indices.max()
    box = (left, top, right + 1, bottom + 1)

    # Crop the image and mask to the bounding box of the mask
    cropped_image = image.crop(box)
    cropped_mask = mask.crop(box)

    return cropped_image, cropped_mask



def pad_to_complete_number_of_patches(image, mask, patch_size=16, bg_color=(245, 245, 245, 255)):
    """ Pad the image and mask to the next multiple of 16 in both dimensions.

    Parameters:
    image (PIL.Image): The image to be padded.
    mask (PIL.Image): The mask to be padded.
    patch_size (int): The size of the patches. Default is 16.
    bg_color (tuple): The background color for padding. Default is (245, 245, 245, 255).
    
    Returns:
    padded_image (PIL.Image): The padded image.
    padded_mask (PIL.Image): The padded mask. """

    w, h = image.size

    # Compute number of patches (round up)
    num_patches_w = math.ceil(w / patch_size)
    num_patches_h = math.ceil(h / patch_size)

    # Make number of patches odd
    if num_patches_w % 2 == 0:
        num_patches_w += 1
    if num_patches_h % 2 == 0:
        num_patches_h += 1

    # Compute target dimensions
    target_w = num_patches_w * patch_size
    target_h = num_patches_h * patch_size

    # Calculate padding
    x_pad = target_w - w
    y_pad = target_h - h
    left = x_pad // 2
    top = y_pad // 2

    # Create new padded image
    padded_image = Image.new("RGBA", (target_w, target_h), bg_color)
    padded_image.paste(image, (left, top), image)

    # Create new padded mask
    padded_mask = Image.new("L", (target_w, target_h), 0)  
    padded_mask.paste(mask, (left, top), mask)
    
    return padded_image, padded_mask



def rotate_image_with_correct_padding(image, mask, angle):
    """ Rotate the image by the given angle and return the rotated image. The background 
    color is set to (245, 245, 245). The image is centered on a square canvas.

    Parameters:
    image (PIL.Image): The image to be rotated.
    mask (PIL.Image): The mask to be rotated.
    angle (float): The angle to rotate the image.

    Returns:
    padded_image (PIL.Image): The rotated image with padding.
    padded_mask (PIL.Image): The rotated mask with padding. """

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    rotated_image = image.rotate(-angle, expand=True, fillcolor=(245, 245, 245, 255))
    rotated_mask = mask.rotate(-angle, expand=True, resample=Image.NEAREST, fillcolor=0)

    # trim the image and mask to remove unnecessary padding
    cropped_image, cropped_mask = crop_to_mask(rotated_image, rotated_mask)

    # Pad the image and mask to the next multiple of 16
    padded_image, padded_mask = pad_to_complete_number_of_patches(cropped_image, cropped_mask)

    # Convert to RGB (remove alpha) and save
    padded_image = padded_image.convert("RGB")

    return padded_image, padded_mask

    

if __name__ == "__main__":

    # Load the configuration
    rotation_info_path = 'data/image_rotations/padding_rotations_ihc.json'
    original_images_path = '../tissue_alignment/data/images/IHC_crops_masked'
    original_masks_path = '../tissue_alignment/data/annotations/IHC_crops'
    rotated_images_path = 'data/IHC_rotated9'
    rotated_masks_path = 'data/IHC_rotated_masks9'

    # Load the rotation info
    with open(rotation_info_path, 'r') as f:
        rotation_info = json.load(f)

    os.makedirs(rotated_images_path, exist_ok=True) # Create the output directory if it doesn't exist
    os.makedirs(rotated_masks_path, exist_ok=True) # Create the output directory for masks if it doesn't exist

    for filename, angle in rotation_info.items(): # Iterate through the dictionary with rotation info
        # Construct the full path to the input and output images
        input_path_image = os.path.join(original_images_path, filename)
        input_path_mask = os.path.join(original_masks_path, filename)
        output_path_image = os.path.join(rotated_images_path, filename)
        output_path_mask = os.path.join(rotated_masks_path, filename)

        # Skip images if marked as "skipped"
        if isinstance(angle, dict) and "skipped" in angle:
            print(f"Skipping {filename} due to 'skipped' flag.") 
            continue
        
        # Rotate the image and save it to the output path
        try:
            img = Image.open(input_path_image)
            mask = Image.open(input_path_mask)
            output_image, output_mask = rotate_image_with_correct_padding(img, mask, float(angle))
            output_image.save(output_path_image)
            output_mask.save(output_path_mask)  

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
