import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
from external.DualInputViT import DualInputViT
from external.DualBranchViT import DualBranchViT
import config


# ------------------------ Dataset ------------------------ #
class MatchPairDataset(Dataset):
    def __init__(self, csv_path, he_dir, ihc_dir):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
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
    
    #print(f"Saved predictions to {output_path}")


# ------------------ Main function ------------------ #
if __name__ == "__main__":
    #set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # define paths
    train_csv = config.train_csv
    val_csv = config.val_csv
    test_csv = config.test_csv

    he_dir = config.he_dir
    ihc_dir = config.ihc_dir

    he_mask_dir = config.he_mask_dir
    ihc_mask_dir = config.ihc_mask_dir

    model_dict_path = config.saved_model_path
    prediction_dir = config.prediction_dir
    os.makedirs(prediction_dir, exist_ok=True)
    output_path = os.path.join(prediction_dir, "predictions.csv")

    device = config.device

    # define model settings
    model_architecture = config.model_architecture
    patch_shape = config.patch_shape
    input_dim = config.input_dim
    embed_dim = config.embed_dim
    n_classes = config.n_classes
    depth = config.depth
    n_heads = config.n_heads
    mlp_ratio = config.mlp_ratio
    load_pretrained_param = config.load_pretrained_param

    if model_architecture == "DualInputViT":
        model = DualInputViT(
            patch_shape = patch_shape, 
            input_dim = input_dim,
            embed_dim = embed_dim, 
            n_classes = n_classes,
            depth = depth,
            n_heads = n_heads,
            mlp_ratio = mlp_ratio,
            pytorch_attn_imp = False,
            init_values = 1e-5,
            act_layer = nn.GELU
        )
    elif model_architecture == "DualBranchViT":
        model = DualBranchViT(
            patch_shape = patch_shape, 
            input_dim = input_dim,
            embed_dim = embed_dim, 
            n_classes = n_classes,
            depth = depth,
            n_heads = n_heads,
            mlp_ratio = mlp_ratio,
            pytorch_attn_imp = False,
            init_values = 1e-5,
            act_layer = nn.GELU
        )

    model.load_state_dict(torch.load(model_dict_path, map_location=device), strict=False)

    model = model.to(device)
    model.eval()

    # Initialize the dataset and dataloader
    eval_dataset = MatchPairDataset(
        csv_path = val_csv,
        he_dir = he_dir,
        ihc_dir = ihc_dir)
    eval_loader = DataLoader(eval_dataset, shuffle=False)

    # Run the model on a dataset
    predictions = run_model(model, eval_loader)

    # Save predictions
    save_predictions(predictions, output_path)