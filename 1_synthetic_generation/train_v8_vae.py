import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from models.vae_v8 import V8VAE
from losses.perceptual_ssim import V8PerceptualLoss
from utils.dataset import OCTDataset

# Configuration Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4        # Lowered learning rate for training stability
LATENT_DIM = 256
BETA = 0.0002               # Tuned KL weight to maintain image sharpness
PERCEPTUAL_WEIGHT = 0.1     # Weight for SSIM & perceptual loss
CHECKPOINT_DIR = "checkpoints_grayscale"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "v8_vae_grayscale_best.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train():
    # 1. Dataset & DataLoaders
    dataset = OCTDataset(data_dir="data")
    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Model, Loss, and Optimizer Setup
    model = V8VAE(latent_dim=LATENT_DIM).to(DEVICE)
    perceptual_fn = V8PerceptualLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_loss = float("inf")

    print(f"🚀 Starting v8 VAE Grayscale Retrain on {DEVICE} for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()

            recon_x, mean, logvar = model(x)

            # L1 Reconstruction Loss for sharp structural boundaries
            recon_loss = F.l1_loss(recon_x, x, reduction="mean")
            
            # KL Divergence Loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            
            # Perceptual & SSIM Loss
            p_ssim_loss = perceptual_fn(recon_x, x)

            # Total Loss Computation
            total_loss = recon_loss + (BETA * kl_loss) + (PERCEPTUAL_WEIGHT * p_ssim_loss)

            total_loss.backward()

            # Gradient Clipping to prevent gradient explosion spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                recon_x, mean, logvar = model(x)

                recon_loss = F.l1_loss(recon_x, x, reduction="mean")
                kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                p_ssim_loss = perceptual_fn(recon_x, x)

                total_loss = recon_loss + (BETA * kl_loss) + (PERCEPTUAL_WEIGHT * p_ssim_loss)
                val_loss += total_loss.item()

        val_loss /= len(val_loader)

        # Save Best Model Weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch [{epoch:03d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print(f"✅ Training Complete! Best model saved to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
