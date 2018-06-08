import torch
import torch.nn as nn


class RecurrentNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, device=None, cell_type="lstm", num_layers=1):
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell_type = cell_type.lower()
        self.num_layers = num_layers
        self.device = device if device != None else torch.device('cpu')

        kwargs = {
            "input_size": self.input_size, 
            "hidden_size": self.hidden_size, 
            "num_layers": self.num_layers,
            "batch_first": True
        }
        
        if self.cell_type == "rnn":
            self.rnn = nn.RNN(**kwargs)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(**kwargs)
        elif self.cell_type == "gru":
            self.rrn = nn.GRU(**kwargs)
        self.fc = nn.Linear(hidden_size, output_size)
            
        self.to(device=device)
            
    
    def forward(self, input_batch, h0):
        batch_size = input_batch.size(0)
        rnn_output, rnn_hidden = self.rnn(input_batch, h0)
        output = self.fc(rnn_output)
        return output, rnn_hidden
    
    def init_hidden(self, batch_size):
        cell_state = torch.empty(self.num_layers, batch_size, self.hidden_size, device=self.device)
        nn.init.xavier_normal_(cell_state)
        
        if self.cell_type == "lstm":
            hidden_state = torch.empty(self.num_layers, batch_size, self.hidden_size, device=self.device)
            nn.init.xavier_normal_(hidden_state)
                  
        if self.cell_type == "lstm":
            return cell_state, hidden_state
        else:
            return hidden_state
        
        
