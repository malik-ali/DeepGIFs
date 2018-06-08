import torch
import torch.nn as nn


class AutoEncoderLSTM(nn.Module):


    # def init_hidden(self, batch_size):
    #     if self.cell_type == "lstm":
    #         return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device), \
    #                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)

    def __init__(self,
                 input_dim,
                 rnn_hidden_size,
                 conv_architecture=None,
                 rnn_cell_type="lstm",
                 num_rnn_layers=1,
                 conv_nonlinearity=nn.ReLU,
                 nonlinearity_every=2,
                 device=None):
        """
        conv_architecture is a list of (num_channels, kernel_size, stride, padding) tuples
        """
        super().__init__()

        self.input_dim = input_dim
        self.encoding_size = AutoEncoderLSTM.get_encoding_size(input_dim, conv_architecture)


        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_cell_type = rnn_cell_type
        self.num_rnn_layers = num_rnn_layers
        self.conv_nonlinearity = conv_nonlinearity
        self.nonlinearity_every=nonlinearity_every

        self.device = device if device != None else torch.device('cpu')

        kwargs = {
            "input_size" : AutoEncoderLSTM.get_encoding_size(self.input_dim, self.conv_architecture),
            "hidden_size" : self.rnn_hidden_size,
            "num_layers": self.num_rnn_layers,
            "batch_first": True
        }


         if self.cell_type == "rnn":
             self.rnn = nn.RNN(**kwargs)
         elif self.cell_type == "lstm":
             self.rnn = nn.LSTM(**kwargs)
         elif self.cell_type == "gru":
             self.rrn = nn.GRU(**kwargs)


         self.to(device=device)


    @staticmethod
    def get_encoding_size(input_dim, conv_architecture):
        """
        input_dim is the shape of the input (N x C x H x W)
        """
        N, C, H, W = input_dim

        assert H == W

        dim = H
        for n_channels, kernel_size, padding, stride in conv_architecture:
            C = n_channels
            dim = (dim + 2 * padding - kernel_size) / stride + 1
            W = H = dim

        return C * H * W

    def make_autoencoder(self):

        _ C, _, _ = self.input_dim
        encoder_layers = []
        decoder_layers = []

        curr_channels = C
        for n, (num_channels, kernel_size, stride, padding) in self.conv_architecture:
            layer_num = n + 1

            encoder_layers.append(nn.Conv2d(in_channels=curr_channels, 
                                            out_channels=num_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding))

            decoder_layers.append(nn.insert(in_channels=num_channels,
                                                     out_channels=curr_channels,
                                                     kernel_size=kernel_size,
                                                     stride=stride), 0)

            if layer_num % self.nonlinearity_every == 0:
                encoder_layers.append(self.conv_nonlinearity())
                decoder_layers.append(self.conv_nonlinearity())

            curr_channels = num_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, input_batch, h0):
        encoded_output = self.encoder(input_batch)
        decoded_output = self.decoder(encoded_output)
        return decoded_output
