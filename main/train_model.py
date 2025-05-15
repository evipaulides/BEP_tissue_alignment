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

# ------------------------ Dataset ------------------------ #
class MultiStainPairDataset(Dataset):
    def __init__(self, csv_path, he_dir, ihc_dir, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        he_name = self.df.loc[idx, "HE"]
        ihc_name = self.df.loc[idx, "IHC"]

        he_path = os.path.join(self.he_dir, he_name)
        ihc_path = os.path.join(self.ihc_dir, ihc_name)

        he_img = Image.open(he_path).convert("RGB")
        ihc_img = Image.open(ihc_path).convert("RGB")

        he_img = TF.to_tensor(he_img)
        ihc_img = TF.to_tensor(ihc_img)

        pos_HE = self.get_positions(he_img, he_path, patch_size=16)
        pos_IHC = self.get_positions(ihc_img, ihc_path, patch_size=16)

        return he_img, ihc_img, pos_HE, pos_IHC
    
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
        

# ------------------------ Model ------------------------ #
class MLPHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

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
        
        plt.imshow(pos_embedding[0, ...])
        plt.show()

        # check if the shape of the features and positional embeddings match
        if x.shape != pos_embedding.shape:
            raise ValueError(
                'Shape of features and positional embedding tensors do not match.',
            )
        # add the combined embedding to each element in the sequence
        x = self.pos_drop(x+pos_embedding)
        
        return x



class MultiStainContrastiveModel(nn.Module):
    def __init__(self, MODEL_NAME, patch_size=16):
        super().__init__()
        self.encoder_he = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        self.encoder_ihc = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        hidden_size = self.encoder_he.embed_dim

        # Patchify images
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=(patch_size),
            stride=(patch_size)
        )

        self.proj_he = MLPHead(in_dim=hidden_size, out_dim=256)
        self.proj_ihc = MLPHead(in_dim=hidden_size, out_dim=256)

    def forward(self, img_he, img_ihc):
        z_he = self.encoder_he(img_he)
        z_ihc = self.encoder_ihc(img_ihc)
        return F.normalize(self.proj_he(z_he), dim=-1), F.normalize(self.proj_ihc(z_ihc), dim=-1)


def contrastive_loss(z1, z2, temperature=0.1):
    sim = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss_a = F.cross_entropy(sim, labels)
    loss_b = F.cross_entropy(sim.T, labels)
    return (loss_a + loss_b) / 2

def model_train(model, optimizer, he, ihc, ACCUMULATION_STEPS, EPOCHS, train_loader, val_loader, device):
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_train = 0
        optimizer.zero_grad()

        for step, (he, ihc) in enumerate(train_loader):
            he, ihc = he.to(device), ihc.to(device)
            z_he, z_ihc = model(he, ihc)
            loss = contrastive_loss(z_he, z_ihc)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            total_train += loss.item()

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        # Average loss for the epoch    
        train_losses.append(total_train / len(train_loader))

        model.eval()
        total_val = 0
        with torch.no_grad():
            for he, ihc in val_loader:
                he, ihc = he.to(device), ihc.to(device)
                z_he, z_ihc = model(he, ihc)
                loss = contrastive_loss(z_he, z_ihc)
                total_val += loss.item()
        val_losses.append(total_val / len(val_loader))

        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

    return model, train_losses, val_losses

# ------------------------ Transform ------------------------ #
class PairTransform:
    def __call__(self, img1, img2):
        # Random horizontal flip
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        # Random vertical flip
        if random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        # Random small rotation (e.g. -5 to +5 degrees)
        angle = random.uniform(-5, 5)
        img1 = TF.rotate(img1, angle, interpolation=TF.InterpolationMode.BILINEAR)
        img2 = TF.rotate(img2, angle, interpolation=TF.InterpolationMode.BILINEAR)

        # Color jitter (same parameters for both images)
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        saturation = random.uniform(0.9, 1.1)
        hue = random.uniform(-0.02, 0.02)

        img1 = TF.adjust_brightness(img1, brightness)
        img2 = TF.adjust_brightness(img2, brightness)

        img1 = TF.adjust_contrast(img1, contrast)
        img2 = TF.adjust_contrast(img2, contrast)

        img1 = TF.adjust_saturation(img1, saturation)
        img2 = TF.adjust_saturation(img2, saturation)

        img1 = TF.adjust_hue(img1, hue)
        img2 = TF.adjust_hue(img2, hue)

        # To tensor and normalize
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        img1 = TF.normalize(img1, mean=[0.5]*3, std=[0.5]*3)
        img2 = TF.normalize(img2, mean=[0.5]*3, std=[0.5]*3)

        return img1, img2


# ------------------------ Main Function ------------------------ #   
if __name__ == "__main__":
    # Config
    EPOCHS = 10
    BATCH_SIZE = 1
    LR = 3e-5
    ACCUMULATION_STEPS = 8
    CHECKPOINT_DIR = "checkpoints"
    RANDOM_SEED = 42
    MODEL_NAME = "vit_base_patch16_224"
    PATCH_SIZE = 16
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load your prepared splits
    train_csv = "data/data_split/train_matches.csv"
    val_csv = "data/data_split/val_matches.csv"

    he_dir = "data/HE_images_matched"
    ihc_dir = "data/IHC_images_matched"

    transform = PairTransform()

    train_data = MultiStainPairDataset(train_csv, he_dir, ihc_dir, transform=transform)
    val_data = MultiStainPairDataset(val_csv, he_dir, ihc_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_idx, (image_HE, image_IHC, pos_HE, pos_IHC) in enumerate(train_loader):
        if batch_idx < 2:
            continue  # stop after a few batches
        if batch_idx > 2:
            break

        image = image_HE.squeeze(0).permute(1, 2, 0).numpy()  # Convert to numpy array

        plt.imshow(image)
        plt.axis('off')
        plt.show()
        print("pos_HE", pos_HE, pos_HE.shape)

    model = MultiStainContrastiveModel(MODEL_NAME, PATCH_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Training
    model, train_losses, val_losses = model_train(model, optimizer, he_dir, ihc_dir, ACCUMULATION_STEPS, EPOCHS, train_loader, val_loader, device)

    # Save model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model.pth"))
    # Save model architecture
    torch.save(model, os.path.join(CHECKPOINT_DIR, "model_architecture.pth"))
    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(CHECKPOINT_DIR, "optimizer.pth"))
    # Save training and validation losses
    torch.save(train_losses, os.path.join(CHECKPOINT_DIR, "train_losses.pth"))
    torch.save(val_losses, os.path.join(CHECKPOINT_DIR, "val_losses.pth"))


    # Save model parts
    torch.save(model.encoder_he.state_dict(), os.path.join(CHECKPOINT_DIR, "vit_he.pth"))
    torch.save(model.encoder_ihc.state_dict(), os.path.join(CHECKPOINT_DIR, "vit_ihc.pth"))
    torch.save(model.proj_he.state_dict(), os.path.join(CHECKPOINT_DIR, "proj_he.pth"))
    torch.save(model.proj_ihc.state_dict(), os.path.join(CHECKPOINT_DIR, "proj_ihc.pth"))

    # Save loss plot
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_plot.png"))
    plt.show()
