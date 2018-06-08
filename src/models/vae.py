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

class Initializer:
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

        model.apply(weights_init)

class VAE(nn.Module):
    def __init__(self, device=None, latent_size=100, num_channels=3):
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

#            nn.Linear(256* 4 * 4, self.latent_size),
#            nn.BatchNorm1d(self.latent_size),
#            nn.LeakyReLU(),
#

#            nn.Linear(128 * 4 * 4, 100)
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, self.latent_size)
        self.fc_logvar = nn.Linear(256 * 4 * 4, self.latent_size)

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
#            nn.Sigmoid()
        )
        
        self.to(device=device)

    def sample_latent(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def get_latent_z(self, x):
        enc_y = self.encoder(x)
        mu, logvar = self.fc_mu(enc_y), self.fc_logvar(enc_y)
        z = self.sample_latent(mu, logvar) 
        return z

    def forward(self, x):
        enc_y = self.encoder(x)
        mu, logvar = self.fc_mu(enc_y), self.fc_logvar(enc_y)
        z = self.sample_latent(mu, logvar) 
        y = self.decoder(z)
        return y, mu, logvar     # uncomment if needed
