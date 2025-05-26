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
from matching_model import MatchingModel


# ------------------------ Dataset ------------------------ #
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
        return len(self.df) * 2  # Half match, half non-match

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

# ------------------------ Run model ------------------------ #
def run_model(model, data_loader):
    predictions = []
    with torch.no_grad():
        for idx, (he_img, he_pos, ihc_img, ihc_pos, label) in enumerate(data_loader):
            # if idx > 30:
            #     break
            #print(f"Step {idx}/{len(data_loader)}")
            he_img, ihc_img = he_img.to(device), ihc_img.to(device)
            he_pos, ihc_pos = he_pos.to(device), ihc_pos.to(device)
            label = label.unsqueeze(1).float().to(device)

            pred = model(he_img, he_pos, ihc_img, ihc_pos)
            prob = F.sigmoid(pred).item()
            predicted = (F.sigmoid(pred) > 0.5).float()
            

            predictions.append({
                "index": idx,
                "true_label": label,
                "pred_prob": prob,
                "pred_label": predicted,
            })

    return predictions

# ------------------- Save predictions ------------------- #
def save_predictions(predictions, output_path):
    output_df = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    #print(f"✅ Saved predictions to {output_path}")


# ------------------ Main function ------------------ #
if __name__ == "__main__":
   # Config
    EPOCHS = 5
    BATCH_SIZE = 1
    LR = 3e-5
    ACCUMULATION_STEPS = 2
    CHECKPOINT_DIR = "checkpoints"
    RANDOM_SEED = 42
    MODEL_NAME = "vit_base_patch16_224"
    PATCH_SIZE = 16
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load your prepared splits
    train_csv = "data/data_split/train_filtered.csv"
    val_csv = "data/data_split/val_filtered.csv"
    test_csv = "data/data_split/test_matches.csv"

    he_dir = "data/HE_images_matched"
    ihc_dir = "data/IHC_images_matched"

     # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MatchingModel(
        model_path="main/external/vit_wee_patch16_reg1_gap_256.sbb_in1k.pth",
        patch_shape=PATCH_SIZE,
        input_dim=3,
        embed_dim=256,
        n_classes=None,
        depth=14,
        n_heads=4,
        mlp_ratio=5,
        pytorch_attn_imp=False,
        init_values=1e-5,
        act_layer=nn.GELU)

    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "matching_model.pth")), map_location=device)
    model = model.to(device)
    model.eval()

    # test_dataset = MatchPairDataset(test_csv_path, he_dir, ihc_dir)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    eval_dataset = MatchPairDataset(val_csv, he_dir, ihc_dir)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Run the model on a dataset
    predictions = run_model(model, eval_loader)

    # Save predictions
    output_path = os.path.join(CHECKPOINT_DIR, "predictions.csv")
    save_predictions(predictions, output_path)

