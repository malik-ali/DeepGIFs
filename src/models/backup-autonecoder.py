import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.device = device if device != None else torch.device('cpu')

        # 3 x 32 x 32
        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 32 x 16 x 16
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64 x 8 x 8 
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 x 4 x 4
            nn.BatchNorm2d(128),
            nn.PReLU(),
#            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 256 x 4 x 4
#            nn.BatchNorm2d(256),
#            nn.PReLU(),
#            nn.Linear(128 * 4 * 4, 100)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),    # 128, 8, 8
            nn.Conv2d(128, 64, 3, stride=1, padding=1),     # 64, 8, 8
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),    # 64, 16, 16
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # 32, 16, 16
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),    # 64, 32, 32
            nn.Conv2d(32, 3, 3, stride=1, padding=1),  # 3, 32, 32
        )
        
        self.to(device=device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


