import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import random
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import random
import math
import numpy as np
from matching_model import MatchingModel
from aug_rotate import rotate_image_with_correct_padding, get_centroid_of_mask

class MatchPairDataset(Dataset):
    def __init__(self, csv_path, he_dir, he_mask_dir, ihc_dir, ihc_mask_dir, transform=False):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.he_mask_dir = he_mask_dir
        self.ihc_mask_dir = ihc_mask_dir
        self.transform = transform
        self.match_map = {
            row['HE']: row['IHC'] for _, row in self.df.iterrows()
        }
        self.he_files = list(self.match_map.keys())
        self.ihc_files = list(self.match_map.values())

    def __len__(self):
        return len(self.df) *2  # Half match, half non-match

    def __getitem__(self, idx):
        is_match = idx % 2 == 0

        if is_match:
            he_name = self.he_files[idx // 2]
            ihc_name = self.match_map[he_name]
            label = 1
        else:
            he_name = random.choice(self.he_files)
            ihc_name = random.choice([f for f in self.ihc_files if f != self.match_map[he_name]])
            label = 0

        he_path = os.path.join(self.he_dir, he_name)
        ihc_path = os.path.join(self.ihc_dir, ihc_name)
        he_mask_path = os.path.join(self.he_mask_dir, he_name)
        ihc_mask_path = os.path.join(self.ihc_mask_dir, ihc_name)

        he_img = Image.open(he_path).convert("RGB")
        ihc_img = Image.open(ihc_path).convert("RGB")
        he_mask = Image.open(he_mask_path).convert("L")
        ihc_mask = Image.open(ihc_mask_path).convert("L")

        #print(f"HE Image: {he_name}, IHC Image: {ihc_name}")


        if self.transform:
            he_img, he_mask, ihc_img, ihc_mask = self.apply_augmentation(he_img, he_mask, ihc_img, ihc_mask)
            
        he_img = TF.to_tensor(he_img)
        ihc_img = TF.to_tensor(ihc_img)

        he_pos = self.get_positions(he_img, he_mask, patch_size=16)
        ihc_pos = self.get_positions(ihc_img, ihc_mask, patch_size=16)

        return he_img, he_pos, ihc_img, ihc_pos, label
    
    def get_positions(self, img, mask, patch_size=16):
        """ Returns the position of each patch in the image. The position is calculated based on the centroid of the mask.

        Parameters:
            img (torch.Tensor): Image for position matrix needs to be calculated.
            mask (PIL.Image): Mask for the image.
            patch_size (int): Size of the patches. Default is 16.
        
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
    
    def augment_image(self, img, mask):
        # Adjust brightness, contrast, saturation and hue
        transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        img = transform(img) # Apply color jitter to image

        # Apply random rotation 
        angle = np.random.random() * 10 -5 # Random rotation angle between 0 and 360 degrees
        img, mask = rotate_image_with_correct_padding(img, mask, -float(angle)) # Rotate image and mask with correct padding
        
        return img, mask
    
    def apply_augmentation(self, he_img, he_mask, ihc_img, ihc_mask):
        # Randomly select augmentation target
        choice = random.choice(['none', 'both', 'HE', 'IHC'])

        #print(f"Applying augmentation: {choice}")
        if choice == 'both':
            he_img, he_mask = self.augment_image(he_img, he_mask)
            ihc_img, ihc_mask = self.augment_image(ihc_img, ihc_mask)
        elif choice == 'HE':
            he_img, he_mask = self.augment_image(he_img, he_mask)
        elif choice == 'IHC':
            ihc_img, ihc_mask = self.augment_image(ihc_img, ihc_mask)
        
        return he_img, he_mask, ihc_img, ihc_mask
    

def train_model(model, train_loader, val_loader, optimizer, device, epochs, accumulation_steps):
    model.to(device)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()

        train_loss = 0
        for step, (he_img, he_pos, ihc_img, ihc_pos, label) in enumerate(train_loader):
            #print(f"Epoch {epoch + 1}/{epochs}, Step {step + 1}/{len(train_loader)}")

            if step > 10:
                break

            step += 1
            he_img, he_pos = he_img.to(device), he_pos.to(device)
            ihc_img, ihc_pos = ihc_img.to(device), ihc_pos.to(device)
            label = label.unsqueeze(1).float().to(device)

            # plot matches ans label
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(he_img[0].cpu().permute(1, 2, 0))
            # plt.title("HE Image")
            # plt.subplot(1, 2, 2)
            # plt.imshow(ihc_img[0].cpu().permute(1, 2, 0))
            # plt.title("IHC Image")
            # plt.suptitle(f"Label: {label.item()}")
            # plt.show()
            pred = model(he_img, he_pos, ihc_img, ihc_pos)
            loss = F.binary_cross_entropy(F.sigmoid(pred), label) / accumulation_steps
            loss.backward()
            train_loss += loss.item()* accumulation_steps

            #print('step:',step, 'label:', label[0].item(), 'Prediction:', F.sigmoid(pred)[0].item(), train_loss)

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
        avg_train_loss = train_loss / step
        train_losses.append(avg_train_loss)
        
        #Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for step, (he_img, he_pos, ihc_img, ihc_pos, label) in enumerate(val_loader):
                if step > 2:
                    break
                he_img, he_pos = he_img.to(device), he_pos.to(device)
                ihc_img, ihc_pos = ihc_img.to(device), ihc_pos.to(device)
                label = label.unsqueeze(1).float().to(device)

                pred = model(he_img, he_pos, ihc_img, ihc_pos)
                loss = F.binary_cross_entropy(F.sigmoid(pred), label)
                val_loss += loss.item()

                predicted = (F.sigmoid(pred) > 0.5).float()
                correct += (predicted == label).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model, train_losses, val_losses

# ------------------------ Plotting Function ------------------------ #
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('plots/model_losses9.png')
    plt.show()



if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Configurations
    EPOCHS = 25
    BATCH_SIZE = 1
    LR = 3e-4
    CHECKPOINT_DIR = "checkpoints"
    RANDOM_SEED = 42
    MODEL_PATH = "main/external/vit_wee_patch16_reg1_gap_256.sbb_in1k.pth"
    PATCH_SIZE = 16
    ACCUMULATION_STEPS = 2
    MODEL_NAME = "matching_model"

    INPUT_DIM = 3
    EMBED_DIM = 256
    DEPTH = 14
    N_HEADS = 4
    MLP_RATIO = 5
    INIT_VALUES = 1e-5
    ACT_LAYER = nn.GELU

    # Paths to data
    train_csv = "data/data_split/train_filtered_copy.csv"
    val_csv = "data/data_split/val_filtered_copy.csv"

    he_dir = "data/HE_images_matched"
    he_mask_dir = "data/HE_masks_matched"
    ihc_dir = "data/IHC_images_matched"
    ihc_mask_dir = "data/IHC_masks_matched"


    # Create datasets and dataloaders
    train_dataset = MatchPairDataset(train_csv, he_dir,  he_mask_dir, ihc_dir, ihc_mask_dir, transform=False)
    val_dataset = MatchPairDataset(val_csv, he_dir, he_mask_dir, ihc_dir, ihc_mask_dir, transform=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = MatchingModel(model_path=MODEL_PATH,
                          patch_shape=PATCH_SIZE,
                          input_dim=INPUT_DIM,
                          embed_dim=EMBED_DIM,
                          n_classes=None,
                          depth=DEPTH,
                          n_heads=N_HEADS,
                          mlp_ratio=MLP_RATIO,
                          pytorch_attn_imp=False,
                          init_values=INIT_VALUES,
                          act_layer=ACT_LAYER
                          )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model, train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device, EPOCHS, ACCUMULATION_STEPS)

    plot_losses(train_losses, val_losses)