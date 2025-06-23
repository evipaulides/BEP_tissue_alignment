import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from PIL import ImageDraw


class PairDataset(Dataset):
    
    def __init__(
        self, 
        csv_path, 
        he_dir, 
        he_mask_dir, 
        ihc_dir, 
        ihc_mask_dir, 
        patch_size=16,
        match_prob=0.5,
        transform=False,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.he_mask_dir = he_mask_dir
        self.ihc_mask_dir = ihc_mask_dir
        self.patch_size = patch_size
        self.transform = transform
        self.match_prob = match_prob
        self.match_map = {
            row['HE']: row['IHC'] for _, row in self.df.iterrows()
        }
        self.he_files = list(self.match_map.keys())
        self.ihc_files = list(self.match_map.values())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        he_name = self.he_files[idx]
        if random.random() < self.match_prob:
            ihc_name = self.match_map[he_name]
            label = 1.0
        else:
            idxs = list(range(len(self.df)))
            idxs.remove(idx)
            ihc_name = self.match_map[self.he_files[random.choice(idxs)]]
            label = 0.0

        he_path = os.path.join(self.he_dir, he_name)
        ihc_path = os.path.join(self.ihc_dir, ihc_name)
        he_mask_path = os.path.join(self.he_mask_dir, he_name)
        ihc_mask_path = os.path.join(self.ihc_mask_dir, ihc_name)

        he_img = Image.open(he_path).convert("RGB")
        ihc_img = Image.open(ihc_path).convert("RGB")
        he_mask = Image.open(he_mask_path).convert("L")
        ihc_mask = Image.open(ihc_mask_path).convert("L")

        if self.transform:
            he_img, he_mask, ihc_img, ihc_mask = self.apply_augmentation(he_img, he_mask, ihc_img, ihc_mask)
            
        he_img = TF.to_tensor(he_img)
        ihc_img = TF.to_tensor(ihc_img)

        he_pos = self.get_positions(he_img, he_mask, patch_size=self.patch_size)
        ihc_pos = self.get_positions(ihc_img, ihc_mask, patch_size=self.patch_size)

        return he_img, he_pos, ihc_img, ihc_pos, label

    
    def get_positions(self, img, mask, patch_size):
        """ Returns the position of each patch in the image. The position is calculated based on the centroid of the mask.

        Parameters:
            img (torch.Tensor): Image for position matrix needs to be calculated.
            mask (PIL.Image): Mask for the image.
            patch_size (int): Size of the patches.
        
        Returns:
            pos (torch.Tensor): Position matrix of the patches. """
        
        # Calculate centroid and convert to pixel value
        centroid = get_centroid_of_mask(mask) # Get the centroid of the mask
        cx = math.floor(centroid[0])
        cy = math.floor(centroid[1])

        # Get the size of the image and number of patches
        C, H, W = img.shape 
        h_patches = H // patch_size
        w_patches = W // patch_size

        # Convert pixel centroid to patch coords
        centroid_patch_x = int(cx // patch_size)
        centroid_patch_y = int(cy // patch_size)

        x_range = torch.arange(w_patches) - centroid_patch_x
        y_range = -(torch.arange(h_patches) - centroid_patch_y)

        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        pos = torch.stack([y_grid, x_grid], dim=2).reshape(-1,2)  # Stack into (H, W, 2) adn then flatten to (H*W, 2)

        return pos
    
    def add_random_background_patches(self, img, mask,  
                                   patch_size_ratio_range=(0.3, 0.7), 
                                   background_color=(245, 245, 245)):
        """
        Adds random background-color patches to simulate partial tissue loss.

        Args:
            img (PIL.Image): The input RGB image.
            mask (PIL.Image): The binary mask (same size as image).
            max_patches (int): Max number of patches to add.
            patch_size_ratio_range (tuple): (min_ratio, max_ratio) for width/height relative to image.
            background_color (tuple): RGB background color.

        Returns:
            img, mask (PIL.Image, PIL.Image): Augmented image and mask.
        """
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        w, h = img.size
        pw = int(random.uniform(*patch_size_ratio_range) * w)
        ph = int(random.uniform(*patch_size_ratio_range) * h)
        px = random.choice([0, w - pw])
        py = random.choice([0, h - ph])

        # Apply patch
        draw_img.rectangle([px, py, px + pw, py + ph], fill=background_color)
        draw_mask.rectangle([px, py, px + pw, py + ph], fill=0)

        return img, mask

    def augment_image(self, img, mask):
        # Adjust brightness, contrast, saturation and hue
        transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        img = transform(img) # Apply color jitter to image

        # Apply random rotation 
        angle = np.random.random() * 10 -5 # Random rotation angle between 0 and 360 degrees
        img, mask = rotate_image_with_correct_padding(img, mask, -float(angle)) # Rotate image and mask with correct padding
        
        # Add random background patches
        if random.random() < 0.3:  # 30% chance to add background patches
            img, mask = self.add_random_background_patches(img, mask, patch_size_ratio_range=(0.3, 0.7), background_color=(245, 245, 245))

        return img, mask
    
    def apply_augmentation(self, he_img, he_mask, ihc_img, ihc_mask):
        # Randomly select augmentation target
        choice = random.choice(['none', 'both', 'HE', 'IHC'])
        
        if choice == 'both':
            he_img, he_mask = self.augment_image(he_img, he_mask)
            ihc_img, ihc_mask = self.augment_image(ihc_img, ihc_mask)
        elif choice == 'HE':
            he_img, he_mask = self.augment_image(he_img, he_mask)
        elif choice == 'IHC':
            ihc_img, ihc_mask = self.augment_image(ihc_img, ihc_mask)
        
        return he_img, he_mask, ihc_img, ihc_mask
    

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


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # define paths
    train_csv = r"C:\Users\rlucasse\OneDrive - UMC Utrecht\Documents\UMCU\projects\debug_matching\train_filtered.csv"
    val_csv = r"C:\Users\rlucasse\OneDrive - UMC Utrecht\Documents\UMCU\projects\debug_matching\val_filtered.csv"

    he_dir = r"C:\Users\rlucasse\OneDrive - UMC Utrecht\Documents\UMCU\projects\debug_matching\HE_images_matched"
    he_mask_dir = r"C:\Users\rlucasse\OneDrive - UMC Utrecht\Documents\UMCU\projects\debug_matching\HE_masks_matched"
    ihc_dir = r"C:\Users\rlucasse\OneDrive - UMC Utrecht\Documents\UMCU\projects\debug_matching\IHC_images_matched"
    ihc_mask_dir = r"C:\Users\rlucasse\OneDrive - UMC Utrecht\Documents\UMCU\projects\debug_matching\IHC_masks_matched"

    train_dataset = PairDataset(
        csv_path=train_csv,
        he_dir=he_dir,
        he_mask_dir=he_mask_dir,
        ihc_dir=ihc_dir,
        ihc_mask_dir=ihc_mask_dir,
        transform=False,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True)

    for img, pos, label in train_dataloader:
        print(label)
        plt.imshow(img[0, ...].numpy().transpose((1,2,0)))
        plt.show()