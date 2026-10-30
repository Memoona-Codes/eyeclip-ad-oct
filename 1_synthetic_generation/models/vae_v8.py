import torch
import torch.nn as nn


class V8VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder (1 Channel Input -> 224x224) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),   # -> (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# -> (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),# -> (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(512 * 14 * 14, latent_dim)
        self.fc_var = nn.Linear(512 * 14 * 14, latent_dim)

        # --- Decoder (256 Latent Dim -> 1 Channel Output 224x224) ---
        self.decoder_input = nn.Linear(latent_dim, 512 * 14 * 14)

        # Replaced ConvTranspose2d with Upsample(nearest) + Conv2d to eliminate grid artifacts
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), # -> (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), # -> (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # -> (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),    # -> (1, 224, 224)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)

        mu = self.fc_mu(features)
        logvar = torch.clamp(self.fc_var(features), min=-6.0, max=6.0)

        z = self.reparameterize(mu, logvar)

        dec_features = self.decoder_input(z)
        dec_features = dec_features.view(-1, 512, 14, 14)
        recon_x = self.decoder(dec_features)

        return recon_x, mu, logvar

    def decode(self, z):
        dec_features = self.decoder_input(z)
        dec_features = dec_features.view(-1, 512, 14, 14)
        return self.decoder(dec_features)
