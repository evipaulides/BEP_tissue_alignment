import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ------------------------ Dataset ------------------------ #
class MultiStainPairDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        he_img = Image.open(self.df.loc[idx, "he_path"]).convert("RGB")
        ihc_img = Image.open(self.df.loc[idx, "ihc_path"]).convert("RGB")

        if self.transform:
            he_img = self.transform(he_img)
            ihc_img = self.transform(ihc_img)

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


# ------------------------ Main Function ------------------------ #
if __name__ == "__main__":
    # Config
    EPOCHS = 10
    BATCH_SIZE = 8
    LR = 3e-5
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) 

    # Data
    df = pd.read_csv("dataset/pairs.csv")  # Make sure this file exists!
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_data = MultiStainPairDataset(train_df, transform=transform)
    val_data = MultiStainPairDataset(val_df, transform=transform)

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
        for he, ihc in train_loader:
            he, ihc = he.to(device), ihc.to(device)
            z_he, z_ihc = model(he, ihc)
            loss = contrastive_loss(z_he, z_ihc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()
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

