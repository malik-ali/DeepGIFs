import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

class Autoencoder(nn.Module):
    def __init__(self, latent_size, device=None, num_channels=3):
        super().__init__()

        self.device = device if device != None else torch.device('cpu')
        self.latent_size = latent_size

        # 3 x 64 x 64
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, stride=2, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64 x 16 x 16 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),


            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            Flatten(),      # 256 * 4 * 4

            nn.Linear(256 * 4 * 4, self.latent_size)
        )


        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 256 * 4 * 4),
#            nn.BatchNorm1d(self.latent_size),
#            nn.PReLU(),

#            nn.Linear(self.latent_size, 128 * 8 * 8),
#            nn.BatchNorm1d(128 * 8 * 8),
#            nn.LeakyReLU(),
            Unflatten(C=256, H=4, W=4),

            nn.Upsample(scale_factor=2, mode='nearest'),    # 256, 8, 8
            nn.Conv2d(256, 128, 3, stride=1, padding=1),     # 128, 8, 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),    # 128, 16, 16
            nn.Conv2d(128, 64, 3, stride=1, padding=1),      # 64, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),    # 64, 32, 32
            nn.Conv2d(64, 32, 3, stride=1, padding=1),      # 32, 32, 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),    # 32, 64, 64
            nn.Conv2d(32, num_channels, 3, stride=1, padding=1),  # 3, 64, 64
            nn.Tanh()
        )
 
        self.to(device=device)

    def forward(self, x):
        enc_y = self.encoder(x)
        y = self.decoder(enc_y)

        return y


