import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

TOTAL_TILES = 24
class LocalPositionalEncoding(nn.Module):
    def __init__(self, model_dim):
        super(LocalPositionalEncoding, self).__init__()
        pe = torch.zeros(TOTAL_TILES * 2, model_dim)
        position = torch.arange(0, TOTAL_TILES * 2, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # return x + self.pe[:, :x.shape[1]].requires_grad_(False)
        return x + self.pe[:, :x.shape[1]]

class LocalTransformerNonSeq(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers):
        super(LocalTransformerNonSeq, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(TOTAL_TILES+1, model_dim)
        self.positional_encoding = LocalPositionalEncoding(model_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
    
    def forward(self, ori_input):
        """
        The shape of the ori_input is now (B, 2, TOTAL_TILES)
        There is no sequence length in this case
        """

        ori_input = ori_input.view(-1, 2*TOTAL_TILES)
        ori_input_embedding = self.embedding(ori_input)
        ori_input_embedding = self.positional_encoding(ori_input_embedding)
        output = self.decoder(ori_input_embedding)
        output = output.view(-1, 2 * TOTAL_TILES * self.model_dim)

        return output

class Nx_chasenet(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers):
        super(Nx_chasenet, self).__init__()
        self.local_transformer = LocalTransformerNonSeq(model_dim, num_heads, num_layers)
        self.linearN = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(2 * TOTAL_TILES * model_dim, 1)),
        ]))

    def forward(self, xs_ori, xs_tar):
        ori_input = torch.cat((xs_ori.unsqueeze(1), xs_tar.unsqueeze(1)), dim=1)
        features = self.local_transformer(ori_input)
        final_output = self.linearN(features)

        return final_output


class Nx_Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Nx_Network, self).__init__()
        self.seq_prepare_layers = nn.Sequential(
            nn.Embedding(in_dim + 1, 640),
            nn.Flatten(),
            nn.Linear(in_dim * 640, 2560),
            nn.BatchNorm1d(2560),
            nn.ReLU(),
            nn.Linear(2560, 2560),
            nn.BatchNorm1d(2560),
            nn.ReLU(),
        )

        self.combined_recover_layers = nn.Sequential(
            nn.Linear(2560 * 2, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),
            nn.Linear(1280, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, out_dim),
        )
    
    def forward(self, xs_ori, xs_tar):
        if isinstance(xs_ori, np.ndarray):
            xs_ori = torch.tensor(xs_ori, dtype=torch.long)
        
        if isinstance(xs_tar, np.ndarray):
            xs_tar = torch.tensor(xs_tar, dtype=torch.long)
        
        seq_prepare_out_ori = self.seq_prepare_layers(xs_ori)
        seq_prepare_out_tar = self.seq_prepare_layers(xs_tar)
        combined_input = torch.cat((seq_prepare_out_ori, seq_prepare_out_tar), dim=1)
        combined_recover_out = self.combined_recover_layers(combined_input)

        return combined_recover_out


class PPO_Actor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PPO_Actor, self).__init__()
        # add a embedding layer to transform the descrete input to continous vectors, considering that the input dim is 6
        self.seq_prepare_layers = nn.Sequential(
            nn.Embedding(in_dim + 1, 512),
            nn.Flatten(),
            nn.Linear(in_dim * 512, 2560),
            
            nn.ReLU(),
            nn.Linear(2560, 2560),
            
            nn.ReLU(),
            nn.Linear(2560, 1280),
            
            nn.ReLU(),
            nn.Linear(1280, 640),
            
            nn.ReLU(),
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            
        )
        

    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.long)

        # write a new forward function with embedding layers considered
        output = self.seq_prepare_layers(obs.long())

        return output
    
class PPO_Critic(nn.Module):
    def __init__(self, in_dim):
        super(PPO_Critic, self).__init__()
        # add a embedding layer to transform the descrete input to continous vectors, considering that the input dim is 6
        self.seq_prepare_layers = nn.Sequential(
            nn.Embedding(in_dim + 1, 512),
            nn.Flatten(),
            nn.Linear(in_dim * 512, 2560),
            
            nn.ReLU(),
            nn.Linear(2560, 2560),
            
            nn.ReLU(),
            nn.Linear(2560, 1280),
            
            nn.ReLU(),
            nn.Linear(1280, 640),
            
            nn.ReLU(),
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            
        )
        

    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.long)

        # write a new forward function with embedding layers considered
        output = self.seq_prepare_layers(obs.long())

        return output
    
