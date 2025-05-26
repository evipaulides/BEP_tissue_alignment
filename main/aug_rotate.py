from PIL import Image, ImageOps
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



def get_centroid_of_mask(mask):
    """ Get the centroid of the mask.

    Parameters:
    mask (PIL.Image): The mask to find the centroid of.

    Returns:
    centroid (tuple): The (x, y) coordinates of the centroid. """

    # Convert to numpy array
    mask_array = np.array(mask)

    # Get coordinates of foreground pixels (non-zero)
    y_indices, x_indices = np.where(mask_array > 0)

    # Compute centroid
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Mask is empty — no foreground pixels found.")

    # Compute centroid
    x_centroid = x_indices.mean()
    y_centroid = y_indices.mean()
    centroid = (x_centroid, y_centroid)
    return centroid



def pad_to_fit_patches_with_centroid(image, mask, centroid, patch_size=16, bg_color=(245, 245, 245, 255)):
    """ Pad the image to patchifyable size, such the the centroid is in the middle of patch (0,0). Since there is no
    center pixel in the patch (16, 16), the centroid is pixel (8,8) (right bottom of 4 center pixels).
    
    Parameters:
        image (PIL.Image): The image to be padded.
        mask (PIL.Image): The mask to be padded.
        centroid (tuple): The (x, y) coordinates of the centroid.
        patch_size (int): The size of the patches.
        bg_color (tuple): The background color to use for padding.
    
    Returns:
        padded_image (PIL.Image): The padded image.
        padded_mask (PIL.Image): The padded mask. """

    # get image size
    w, h = image.size

    # Get the centroid coordinates
    cx, cy = centroid

    # Convert to pixel value
    cx = math.floor(cx)
    cy = math.floor(cy)

    # Caclulate target image dimensions
    number_extra_pixels_left = cx % patch_size
    if number_extra_pixels_left < patch_size // 2:
        add_left = patch_size // 2 - number_extra_pixels_left
    else:
        add_left = patch_size + patch_size // 2 - number_extra_pixels_left

    number_extra_pixels_top = cy % patch_size
    if number_extra_pixels_top < patch_size // 2:
        add_top = patch_size // 2 - number_extra_pixels_top
    else:
        add_top = patch_size + patch_size // 2 - number_extra_pixels_top

    number_extra_pixels_right = (w - cx) % patch_size
    if number_extra_pixels_right < patch_size // 2:
        add_right = patch_size // 2 - number_extra_pixels_right
    else:
        add_right = patch_size + patch_size // 2 - number_extra_pixels_right

    number_extra_pixels_bottom = (h - cy) % patch_size
    if number_extra_pixels_bottom < patch_size // 2:
        add_bottom = patch_size // 2 - number_extra_pixels_bottom
    else:
        add_bottom = patch_size + patch_size // 2 - number_extra_pixels_bottom

    # Create new padded image and mask
    padded_image = ImageOps.expand(image, border=(add_left, add_top, add_right, add_bottom), fill=bg_color)
    padded_mask = ImageOps.expand(mask, border=(add_left, add_top, add_right, add_bottom), fill=0)  

    return padded_image, padded_mask



def rotate_image_with_correct_padding(image, mask, angle, bg_color=(245, 245, 245, 255)):
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
    
    # Remove adjusted background from image augmentation
    background = Image.new("RGBA", image.size, bg_color)
    image = Image.composite(image, background, mask)

    # Rotate the image and mask
    rotated_image = image.rotate(-angle, expand=True, fillcolor=bg_color)
    rotated_mask = mask.rotate(-angle, expand=True, resample=Image.NEAREST, fillcolor=0)

    # Trim the image and mask to remove unnecessary padding
    cropped_image, cropped_mask = crop_to_mask(rotated_image, rotated_mask)

    # Get the centroid of the mask
    centroid = get_centroid_of_mask(cropped_mask)

    # # Pad the image and mask to next multiple of 16 with the centroid in the center of a patch
    padded_image, padded_mask = pad_to_fit_patches_with_centroid(cropped_image, cropped_mask, centroid)

    # Convert to RGB (remove alpha) and save
    padded_image = padded_image.convert("RGB")

    return padded_image, padded_mask