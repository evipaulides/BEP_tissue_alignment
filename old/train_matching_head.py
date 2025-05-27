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

# ------------------------ Encoder ------------------------ #
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
# ------------------------                   ------------------------ #

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
    
# ---------------- Dataset for match classification ----------------- #
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
    
# -------------------------- Training --------------------------- #
def train_match_head(model_he, model_ihc, match_head, train_loader,val_loader, optimizer, device, EPOCHS, ACCUMULATION_STEPS):
    model_he.eval()
    model_ihc.eval()
    match_head.train()

    train_losses = []
    val_losses = []

    #print('Start training')

    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()

        #print(f"Epoch {epoch+1}/{EPOCHS}")

        for step, (he_img, he_pos, ihc_img, ihc_pos, labels) in enumerate(train_loader):
            # if step> 1:
            #     break
            #print(f"Step {step+1}/{len(train_loader)}")
            he_img = he_img.to(device)
            ihc_img = ihc_img.to(device)
            he_pos = he_pos.to(device)
            ihc_pos = ihc_pos.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            with torch.no_grad():
                z_he = model_he(he_img, he_pos)
                z_ihc = model_ihc(ihc_img, ihc_pos)

            pred = match_head(z_he, z_ihc)
            loss = F.binary_cross_entropy(F.sigmoid(pred), labels) / ACCUMULATION_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUMULATION_STEPS

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        match_head.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for step, (he_img, he_pos, ihc_img, ihc_pos, labels) in enumerate(val_loader):
                # if step > 1:
                #     break
                #print(f"Step {step+1}/{len(val_loader)}")
                he_img = he_img.to(device)
                ihc_img = ihc_img.to(device)
                he_pos = he_pos.to(device)
                ihc_pos = ihc_pos.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                z_he = model_he(he_img, he_pos)
                z_ihc = model_ihc(ihc_img, ihc_pos)
                pred = match_head(z_he, z_ihc)
                val_loss += F.binary_cross_entropy(pred, labels).item()

                predicted = (pred > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total
        #print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Val Acc = {val_acc:.2%}")

    #print(train_losses, val_losses)
    return match_head, train_losses, val_losses

# ------------------------ Plotting Function ------------------------ #
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

# ------------------------ Save models ------------------------ #	
def save_model(model, checkpoint_dir, model_name):
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{model_name}.pth"))
    #torch.save(model, os.path.join(checkpoint_dir, "model_architecture.pth"))
    #torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pth"))
    # Save losses
    torch.save(train_losses, os.path.join(checkpoint_dir, "head_train_losses.pth"))
    torch.save(val_losses, os.path.join(checkpoint_dir, "head_val_losses.pth"))


# ------------------------ Main Function ------------------------ #
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

    he_dir = "data/HE_images_matched"
    ihc_dir = "data/IHC_images_matched"

     # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_he = MultiStainContrastiveModel(MODEL_NAME, PATCH_SIZE).to(device)
    model_ihc = MultiStainContrastiveModel(MODEL_NAME, PATCH_SIZE).to(device)

    model_he.load_state_dict(torch.load("checkpoints/model_he.pth"))
    model_ihc.load_state_dict(torch.load("checkpoints/model_ihc.pth"))
    model_he.eval()
    model_ihc.eval()

    # Matching head
    match_train_dataset = MatchPairDataset(train_csv, he_dir, ihc_dir)
    match_val_dataset = MatchPairDataset(val_csv, he_dir, ihc_dir)
    match_train_loader = DataLoader(match_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    match_val_loader = DataLoader(match_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    match_head = MatchingHead(embed_dim=256).to(device)
    match_optimizer = torch.optim.AdamW(match_head.parameters(), lr=LR)

    match_head, train_losses, val_losses = train_match_head(model_he, model_ihc, match_head, match_train_loader, match_val_loader, match_optimizer, device, EPOCHS, ACCUMULATION_STEPS)

    # Save the model
    save_model(match_head, CHECKPOINT_DIR, "matching_head")