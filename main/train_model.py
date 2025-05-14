import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision.transforms import functional as TF

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

        if self.transform:
            he_img, ihc_img = self.transform(he_img, ihc_img)

        return he_img, ihc_img

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

class MultiStainContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_he = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.encoder_ihc = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.proj_he = MLPHead()
        self.proj_ihc = MLPHead()

    def forward(self, img_he, img_ihc):
        z_he = self.encoder_he(pixel_values=img_he).last_hidden_state[:, 0]
        z_ihc = self.encoder_ihc(pixel_values=img_ihc).last_hidden_state[:, 0]
        return F.normalize(self.proj_he(z_he), dim=-1), F.normalize(self.proj_ihc(z_ihc), dim=-1)

def contrastive_loss(z1, z2, temperature=0.1):
    sim = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss_a = F.cross_entropy(sim, labels)
    loss_b = F.cross_entropy(sim.T, labels)
    return (loss_a + loss_b) / 2

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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load your prepared splits
    train_csv = "data/data_split/train_matches.csv"
    val_csv = "data/data_split/val_matches.csv"

    he_dir = "data/HE_images_matched"
    ihc_dir = "data/IHC_images_matched"

    transform = PairTransform()

    train_data = MultiStainPairDataset(train_csv, he_dir, ihc_dir, transform=transform)
    val_data = MultiStainPairDataset(val_csv, he_dir, ihc_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStainContrastiveModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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
