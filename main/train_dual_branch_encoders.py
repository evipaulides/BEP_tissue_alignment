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

# ------------------------ Dataset ------------------------ #    
class TripletStainDataset(Dataset):
    """
    Returns: anchor, positive, negative 
    """
    def __init__(self, csv_path, anchor_dir, match_dir, anchor_stain, match_stain, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.anchor_dir = anchor_dir
        self.match_dir = match_dir
        self.transform = transform
        self.anchor_stain = anchor_stain
        self.match_stain = match_stain

        # Build a map from HE image to matched IHC image
        self.match_map = {
            row[self.anchor_stain]: row[self.match_stain] for _, row in self.df.iterrows()
        }

        # All possible match image names
        self.match_pool = self.df[self.match_stain].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor_name = self.df.loc[idx, self.anchor_stain]
        match_pos_name = self.match_map[anchor_name]

        # Ensure negative is not the positive
        match_neg_name = match_pos_name
        while match_neg_name == match_pos_name:
            match_neg_name = random.choice(self.match_pool)

        anchor_path = os.path.join(self.anchor_dir, anchor_name)
        match_pos_path = os.path.join(self.match_dir, match_pos_name)
        match_neg_path = os.path.join(self.match_dir, match_neg_name)

        anchor_img = Image.open(anchor_path).convert("RGB")
        match_pos_img = Image.open(match_pos_path).convert("RGB")
        match_neg_img = Image.open(match_neg_path).convert("RGB")

        
        anchor_img = TF.to_tensor(anchor_img)
        match_pos_img = TF.to_tensor(match_pos_img)
        match_neg_img = TF.to_tensor(match_neg_img)

        # Get positions
        anchor_pos = self.get_positions(anchor_img, anchor_path, patch_size=16)
        match_pos_pos = self.get_positions(match_pos_img, match_pos_path, patch_size=16)
        match_neg_pos = self.get_positions(match_neg_img, match_neg_path, patch_size=16)

        return anchor_img, anchor_pos, match_pos_img, match_pos_pos, match_neg_img, match_neg_pos
    
    def get_positions(self, img, path, patch_size=16):
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
    
# ------------------------ Encoder ------------------------ #
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
    
# ------------------------ Training Function ------------------------ #
def train_dual_encoder(model_he, model_ihc, loader_he, loader_ihc, val_loader_he, val_loader_ihc, optimizer, device, epochs=10, checkpoint_dir=None, accumulation_steps=8):
    model_he.to(device)
    model_ihc.to(device)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model_he.train()
        model_ihc.train()

        total_loss = 0
        optimizer.zero_grad()
        for step, ((he_img, he_pos, ihc_match, ihc_match_pos, ihc_neg, ihc_neg_pos), (ihc_img, ihc_pos, he_match, he_match_pos, he_neg, he_neg_pos)) in enumerate(zip(loader_he, loader_ihc)):
            if step > 1: # For testing purposes, remove this line in production
                break
            print('step', step)
            he_img, ihc_match, ihc_neg, ihc_img, he_match, he_neg = he_img.to(device), ihc_match.to(device), ihc_neg.to(device), ihc_img.to(device), he_match.to(device), he_neg.to(device)
            he_pos, ihc_match_pos, ihc_neg_pos, ihc_pos, he_match_pos, he_neg_pos = he_pos.to(device), ihc_match_pos.to(device), ihc_neg_pos.to(device), ihc_pos.to(device), he_match_pos.to(device), he_neg_pos.to(device)

            z_anchor_he = model_he(he_img, he_pos)        # HE
            z_pos_ihc   = model_ihc(ihc_match, ihc_match_pos)             # IHC (match)
            z_neg_ihc   = model_ihc(ihc_neg, ihc_neg_pos)             # IHC (non-match)


            z_anchor_ihc = model_ihc(ihc_img, ihc_pos)        # HE
            z_pos_he   = model_he(he_match, he_match_pos)             # IHC (match)
            z_neg_he   = model_he(he_neg, he_neg_pos)             # IHC (non-match)

            print('start loss')
            loss_he  = F.triplet_margin_loss(z_anchor_he, z_pos_ihc, z_neg_ihc)
            loss_ihc = F.triplet_margin_loss(z_anchor_ihc, z_pos_he, z_neg_he)
            loss = (loss_he + loss_ihc) / 2
            print('end loss')
            
            loss = loss / accumulation_steps
            
            loss.backward()
            total_loss += loss.item()
            print('backward done')
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader_he):
                optimizer.step()
                optimizer.zero_grad()

        #avg_loss = total_loss / len(loader_he)
        avg_loss = total_loss / step # step+ 1 van maken als de break statement weg gaat !!!
        train_losses.append(avg_loss)

        print('start validation')
        # Validation
        model_he.eval()
        model_ihc.eval()
        val_loss = 0
        with torch.no_grad():
            for step, ((he_img, he_pos, ihc_match, ihc_match_pos, ihc_neg, ihc_neg_pos), (ihc_img, ihc_pos, he_match, he_match_pos, he_neg, he_neg_pos)) in enumerate(zip(val_loader_he, val_loader_ihc)):
                if step > 1:
                    break

                he_img, ihc_match, ihc_neg, ihc_img, he_match, he_neg = he_img.to(device), ihc_match.to(device), ihc_neg.to(device), ihc_img.to(device), he_match.to(device), he_neg.to(device)
                he_pos, ihc_match_pos, ihc_neg_pos, ihc_pos, he_match_pos, he_neg_pos = he_pos.to(device), ihc_match_pos.to(device), ihc_neg_pos.to(device), ihc_pos.to(device), he_match_pos.to(device), he_neg_pos.to(device)

                z_anchor_he = model_he(he_img, he_pos)        # HE
                z_pos_ihc   = model_ihc(ihc_match, ihc_match_pos)             # IHC (match)
                z_neg_ihc   = model_ihc(ihc_neg, ihc_neg_pos)             # IHC (non-match)


                z_anchor_ihc = model_ihc(ihc_img, ihc_pos)        # HE
                z_pos_he   = model_he(he_match, he_match_pos)             # IHC (match)
                z_neg_he   = model_he(he_neg, he_neg_pos)             # IHC (non-match)

                print('start loss')
                loss_he  = F.triplet_margin_loss(z_anchor_he, z_pos_ihc, z_neg_ihc)
                loss_ihc = F.triplet_margin_loss(z_anchor_ihc, z_pos_he, z_neg_he)
                loss = (loss_he + loss_ihc) / 2
                print('end loss')
                val_loss += loss.item()

        avg_val_loss = val_loss / step # step+ 1 van maken als de break statement weg gaat !!!
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print(train_losses, val_losses)
    return model_he, model_ihc, train_losses, val_losses

# ------------------------ Plotting Function ------------------------ #
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1) ,train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid()
    plt.savefig("plots/losses.png")
    plt.show()

# ------------------------ Save models ------------------------ #	
def save_model(model, checkpoint_dir, model_name, train_losses=None, val_losses=None):
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{model_name}.pth"))
    #torch.save(model, os.path.join(checkpoint_dir, "model_architecture.pth"))
    #torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pth"))
    # Save losses
    torch.save(train_losses, os.path.join(checkpoint_dir, "train_losses.pth"))
    torch.save(val_losses, os.path.join(checkpoint_dir, "val_losses.pth"))


# ------------------------ Main Function ------------------------ #   
if __name__ == "__main__":
    # Config
    EPOCHS = 3
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
    # train_HE = MultiStainPairDataset(train_csv, he_dir, "HE")
    # train_IHC = MultiStainPairDataset(train_csv, ihc_dir, "IHC")

    # val_HE = MultiStainPairDataset(val_csv, he_dir, "HE")
    # val_IHC = MultiStainPairDataset(val_csv, ihc_dir, "IHC")

    train_HE = TripletStainDataset(train_csv, he_dir, ihc_dir, "HE", "IHC")
    train_IHC = TripletStainDataset(train_csv, ihc_dir, he_dir, "IHC", "HE")

    val_HE = TripletStainDataset(val_csv, he_dir, ihc_dir, "HE", "IHC")
    val_IHC = TripletStainDataset(val_csv, ihc_dir, he_dir, "IHC", "HE")

    train_loader_HE = DataLoader(train_HE, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_IHC = DataLoader(train_IHC, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_HE = DataLoader(val_HE, batch_size=BATCH_SIZE, shuffle=False)
    val_loader_IHC = DataLoader(val_IHC, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_he = MultiStainContrastiveModel(MODEL_NAME, PATCH_SIZE).to(device)
    model_ihc = MultiStainContrastiveModel(MODEL_NAME, PATCH_SIZE).to(device)

    torchinfo.summary(model_he)

    optimizer = torch.optim.AdamW(list(model_he.parameters()) + list(model_ihc.parameters()), lr=LR)

    model_he, model_ihc = model_he.to(device), model_ihc.to(device)

    print("start training")
    # Training
    model_he, model_ihc, train_losses, val_losses = train_dual_encoder(model_he, model_ihc, train_loader_HE, train_loader_IHC, val_loader_HE, val_loader_IHC, optimizer, device, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR, accumulation_steps=ACCUMULATION_STEPS)
    print("end training")

    # Save the model
    save_model(model_he, CHECKPOINT_DIR, "model_he", train_losses, val_losses)
    save_model(model_ihc, CHECKPOINT_DIR, "model_ihc")
    
    # Plot losses
    plot_losses(train_losses, val_losses)