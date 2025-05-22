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

# ------------------------ Encoders ------------------------ #
class PatchEmbedder(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768, dropout=0.1):
        super(PatchEmbedder, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Convolutional layer to create patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # Shape: (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2, 3).transpose(1, 2) # Shape: (B, num_patches, embed_dim)

        return x

class PositionalEmbedder(nn.Module):
    max_position_index = 100
    repeat_section_embedding = True

    def __init__(self, embed_dim: int, dropout_prob: float):
        super().__init__()
        # initialize instance attributes
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(
            data=torch.zeros((self.max_position_index+1, self.embed_dim//2)), requires_grad=False) # (max_position_index+1, embed_dim//2)
        X = torch.arange(self.max_position_index+1, dtype=torch.float32).reshape(-1, 1) # (max_position_index+1, 1)
        X = X / torch.pow(10000, torch.arange(0, self.embed_dim//2, 2, dtype=torch.float32) / (self.embed_dim//2))
        self.pos_embed[:, 0::2] = torch.sin(X)
        self.pos_embed[:, 1::2] = torch.cos(X)

        # initialize dropout layer
        self.pos_drop = nn.Dropout(p=dropout_prob)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor): # Through the model
        pos = torch.round(pos).to(int)
        if torch.max(pos[:, :, 1:]) > self.max_position_index:
            raise ValueError(
                'Maximum requested position index exceeds the prepared position indices.'
            )
        # get the number of items in the batch and the number of tokens in the sequence
        B, S, _ = pos.shape
        device = self.pos_embed.get_device()
        if device == -1:
            device = 'cpu'

        # define embeddings for x and y dimension
        embeddings = [self.pos_embed[pos[:, :, 0], :],
                      self.pos_embed[pos[:, :, 1], :]]
        # add a row of zeros as padding in case the embedding dimension has an odd length
        if self.embed_dim % 2 == 1:
            embeddings.append(torch.zeros((B, S, 1), device=device))

        # prepare positional embedding
        pos_embedding = torch.concat(embeddings, dim=-1)

        # account for [CLS] token
        pos_embedding = torch.concatenate(
            [torch.zeros((B, 1, self.embed_dim), device=device), pos_embedding], dim=1,
        )
        
        # plt.imshow(pos_embedding[0, ...])
        # plt.show()

        # check if the shape of the features and positional embeddings match
        if x.shape != pos_embedding.shape:
            raise ValueError(
                'Shape of features and positional embedding tensors do not match.',
            )
        # add the combined embedding to each element in the sequence
        x = self.pos_drop(x+pos_embedding)
        
        return x
    
class MLPHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, out_dim)
        ) # Dropout nog toevoegen?

    def forward(self, x):
        return self.net(x)
    
class MultiStainContrastiveModel(nn.Module):
    def __init__(self, MODEL_NAME, patch_size=16, dropout_prob=0.1):	
        super().__init__()
        self.patch_size = patch_size
        self.dropout_prob = dropout_prob

        # Load the pre-trained model
        vit = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        self.embed_dim = vit.embed_dim

        # Define the patch embedding layer
        self.patch_embed = PatchEmbedder(patch_size=patch_size, in_channels=3, embed_dim=self.embed_dim, dropout=dropout_prob)

        # Define the positional embedding layer
        self.pos_embed = PositionalEmbedder(embed_dim=self.embed_dim, dropout_prob=dropout_prob)

        # Reuse the ViT blocks and normalization layer
        self.blocks = vit.blocks
        self.norm = vit.norm

        # initialize cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Define the MLP head
        self.mlp_head = MLPHead(in_dim=self.embed_dim, out_dim=256)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # Patch embedding
        x = self.patch_embed(x)

        # create and add the [CLS] token to the sequence of embeddings 
        # (one for each item in the batch)
        # [1, 1, D] -> [B, 1, D]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # [B, S, D] -> [B, 1+S, D]
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional embedding
        x = self.pos_embed(x, pos)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Normalization
        x = self.norm(x)

        # MLP head
        x = x[:, 0]  # Select the CLS token
        output = self.mlp_head(x)  # Ik weet niet of dit nog aangepast moet worden naar mijn probleem

        return output

# ---------------------- Matching head ----------------------------- #
class MatchingHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z1, z2):
        x = torch.cat([z1, z2], dim=1)  # shape (B, 2D)
        return self.net(x)              # shape (B, 1), match probability
    

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
def run_model(model_he, model_ihc, matching_head, data_loader):
    predictions = []
    with torch.no_grad():
        for idx, (he_img, he_pos, ihc_img, ihc_pos, label) in enumerate(data_loader):
            # if idx > 30:
            #     break
            #print(f"Step {idx}/{len(data_loader)}")
            he_img, ihc_img = he_img.to(device), ihc_img.to(device)
            he_pos, ihc_pos = he_pos.to(device), ihc_pos.to(device)
            label = label.item()

            z_he = model_he(he_img, he_pos)
            z_ihc = model_ihc(ihc_img, ihc_pos)
            prob = matching_head(z_he, z_ihc).item()
            pred = int(prob > 0.5)

            predictions.append({
                "index": idx,
                "true_label": label,
                "pred_prob": prob,
                "pred_label": pred
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
    train_csv = "data/data_split/train_matches.csv"
    val_csv = "data/data_split/val_matches.csv"
    test_csv = "data/data_split/test_matches.csv"

    he_dir = "data/HE_images_matched"
    ihc_dir = "data/IHC_images_matched"

     # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_he = MultiStainContrastiveModel(MODEL_NAME, PATCH_SIZE).to(device)
    model_ihc = MultiStainContrastiveModel(MODEL_NAME, PATCH_SIZE).to(device)
    matching_head = MatchingHead(embed_dim=256).to(device)

    model_he.load_state_dict(torch.load("checkpoints/model_he.pth"))
    model_ihc.load_state_dict(torch.load("checkpoints/model_ihc.pth"))
    matching_head.load_state_dict(torch.load("checkpoints/matching_head.pth"))
    model_he.eval()
    model_ihc.eval()
    matching_head.eval()

    # test_dataset = MatchPairDataset(test_csv_path, he_dir, ihc_dir)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    eval_dataset = MatchPairDataset(val_csv, he_dir, ihc_dir)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Run the model on a dataset
    predictions = run_model(model_he, model_ihc, matching_head, eval_loader)

    # Save predictions
    output_path = os.path.join(CHECKPOINT_DIR, "predictions.csv")
    save_predictions(predictions, output_path)

