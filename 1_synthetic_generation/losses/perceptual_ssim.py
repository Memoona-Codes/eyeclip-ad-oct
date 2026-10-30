import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


def compute_ssim(x, y, window_size=11):
    """Computes Structural Similarity Index (SSIM) between two tensors."""
    C = x.size(1)
    kernel = torch.ones((C, 1, window_size, window_size), device=x.device) / (window_size ** 2)
    
    mu_x = F.conv2d(x, kernel, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2, groups=C)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=window_size // 2, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=window_size // 2, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2, groups=C) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return ssim_map.mean()


class V8PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet50 for perceptual feature matching
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.slice1 = nn.Sequential(*list(resnet.children())[:5])   # Conv1 to Layer1
        self.slice2 = nn.Sequential(*list(resnet.children())[5:6])  # Layer2
        self.slice3 = nn.Sequential(*list(resnet.children())[6:7])  # Layer3

        # Freeze perceptual network parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.eval()

    def forward(self, recon_x, x):
        # 1. Expand single-channel grayscale (1 channel) to 3 channels for ResNet
        if x.size(1) == 1:
            x_in = x.repeat(1, 3, 1, 1)
            recon_in = recon_x.repeat(1, 3, 1, 1)
        else:
            x_in, recon_in = x, recon_x

        # 2. Perceptual Feature Loss
        h1_x = self.slice1(x_in)
        h2_x = self.slice2(h1_x)
        h3_x = self.slice3(h2_x)

        h1_recon = self.slice1(recon_in)
        h2_recon = self.slice2(h1_recon)
        h3_recon = self.slice3(h2_recon)

        perceptual_loss = (
            F.l1_loss(h1_recon, h1_x) +
            F.l1_loss(h2_recon, h2_x) +
            F.l1_loss(h3_recon, h3_x)
        )

        # 3. SSIM Loss (Structural Similarity)
        ssim_val = compute_ssim(recon_x, x)
        ssim_loss = 1.0 - ssim_val

        return perceptual_loss + (0.5 * ssim_loss)
