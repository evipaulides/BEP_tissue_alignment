import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision.transforms import functional as TF
import timm
import random
import numpy as np
import torchinfo
from matching_head import MatchingHead

class MatchPairDataset(Dataset):
    def __init__(self, csv_path, he_dir, ihc_dir, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.transform = transform
        self.match_map = {
            row['HE']: row['IHC'] for _, row in self.df.iterrows()
        }
        self.he_files = list(self.match_map.keys())
        self.ihc_files = list(self.match_map.values())

    def __len__(self):
        return len(self.df) #*2  # Half match, half non-match

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

        he_img = Image.open(he_path).convert("RGB")
        ihc_img = Image.open(ihc_path).convert("RGB")

        if self.transform:
            he_img = self.transform(he_img)
            ihc_img = self.transform(ihc_img)
        else:
            he_img = TF.to_tensor(he_img)
            ihc_img = TF.to_tensor(ihc_img)

        he_pos = self.get_positions(he_img, patch_size=16)
        ihc_pos = self.get_positions(ihc_img, patch_size=16)

        return he_img, he_pos, ihc_img, ihc_pos, label
    
    def get_positions(self, img, patch_size=16):
        C, H, W = img.shape
        # Calculate the number of patches in each dimension
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        # Create grid of patches with 0,0 at center
        grid_h = torch.arange(num_patches_h)
        grid_w = torch.arange(num_patches_w)
        center_h = (num_patches_h - 1) / 2
        center_w = (num_patches_w - 1) / 2
        pos_h = (center_h - grid_h)
        pos_w = (grid_w - center_w)

        # Create position grid of 
        pos = torch.stack(torch.meshgrid(pos_h, pos_w), dim=-1).reshape(-1, 2) # numpy array (1, W*H, 2)

        return pos
    
def train_model(model, train_loader, val_loader, optimizer, device, epochs, accumulation_steps):
    model.to(device)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()

        train_loss = 0
        for step, (he_img, he_pos, ihc_img, ihc_pos, label) in enumerate(train_loader):
            print(f"Epoch {epoch + 1}/{epochs}, Step {step + 1}/{len(train_loader)}")
            # if step > 10:
            #     break
            step += 1
            he_img, he_pos = he_img.to(device), he_pos.to(device)
            ihc_img, ihc_pos = ihc_img.to(device), ihc_pos.to(device)
            label = label.unsqueeze(1).float().to(device)

            pred = model(he_img, he_pos, ihc_img, ihc_pos)
            loss = F.binary_cross_entropy(F.sigmoid(pred), label) / accumulation_steps
            loss.backward()
            train_loss += loss.item()* accumulation_steps

            print('step:',step, 'label:', label[0].item(), 'Prediction:', F.sigmoid(pred)[0].item(), train_loss)

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                print(f"Step {step + 1}/{len(train_loader)}, Loss: {train_loss / step:.4f}")
            
        avg_train_loss = train_loss / step
        train_losses.append(avg_train_loss)
        


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Configurations
    EPOCHS = 10
    BATCH_SIZE = 1
    LR = 3e-5
    CHECKPOINT_DIR = "checkpoints"
    RANDOM_SEED = 42
    MODEL_PATH = "main/external/vit_wee_patch16_reg1_gap_256.sbb_in1k.pth"
    PATCH_SIZE = 16
    ACCUMULATION_STEPS = 2

    # Paths to data
    train_csv = "data/data_split/train_matches - Copy.csv"
    val_csv = "data/data_split/val_matches.csv"

    he_dir = "data/HE_images_matched"
    ihc_dir = "data/IHC_images_matched"

    # Create datasets and dataloaders
    train_dataset = MatchPairDataset(train_csv, he_dir, ihc_dir)
    val_dataset = MatchPairDataset(val_csv, he_dir, ihc_dir)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = MatchingHead(embed_dim=256, model_path=MODEL_PATH, act_layer=nn.GELU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(torchinfo.summary(model))

    #train_model(model, train_loader, val_loader, optimizer, device, EPOCHS, ACCUMULATION_STEPS)

    