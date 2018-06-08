import torch
import torch.nn as nn

from .recurrent_network import RecurrentNetwork
from .autoencoder import Autoencoder

class LatentLSTM(nn.Module):
    
    def __init__(self, latent_size, device=None, num_channels=3, cell_type="lstm", num_rnn_layers=1):
        
        super().__init__()
        
        self.stats = locals()

        device = device if device != None else torch.device('cpu')

        self.ae = Autoencoder(latent_size=latent_size,
                              device=device, 
                              num_channels=num_channels)

        self.rnn = RecurrentNetwork(input_size=latent_size, 
                                    hidden_size=latent_size,
                                    output_size=latent_size,
                                    device=device,
                                    cell_type=cell_type,
                                    num_layers=num_rnn_layers)
        
        self.to(device=device)
    
    def forward(self, input_batch, h0):
        """
            - input_batch:  Shape: N x T x C x H x W 
        """

#         latent_zs = self.get_latent_zs(input_batch)   # latent_zs: N x T x L
        N, T, C, H, W = input_batch.shape
        input_batch = input_batch.view((N*T, C, H, W))    # NT x C x H x W
        zs = self.ae.encoder(input_batch)                 # NT x L
        zs = zs.view((N, T, self.ae.latent_size))         # N x T x L
        next_zs, next_h = self.rnn(zs, h0)                # N x T x L
        next_zs = next_zs.contiguous().view((N*T, -1))    # NT X L
        ys = self.ae.decoder(next_zs)                     # NT x C x H x W
        ys = ys.view((N, T, C, H, W))                     # N x T x C x H x W
        
        return ys, next_h

    
    def get_latent_zs(self, x):
        """
            - x has shape N x T x C x H x W
            - encoder/decoder take N x C x H x W  --> N x L
            - returns N x T x L
        """
        # Encoder expects N x C x H x W
        N, T, C, H, W = x.shape
        x = x.view((N*T, C, H, W))
        enc_y = self.vae.encoder(x)
        mu = self.vae.fc_mu(enc_y)
        logvar = self.vae.fc_logvar(enc_y)
        ret =  self.vae.sample_latent(mu, logvar)
        return ret.view((N, T, self.vae.latent_size))
    
    
    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)
