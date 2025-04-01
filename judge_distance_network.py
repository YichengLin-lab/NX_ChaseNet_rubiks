import torch
from torch import nn
import torch.nn.functional as F
import numpy as np




# define a function that can convert an integer to a one-hot representation
def one_hot(x, class_count):
    return torch.eye(class_count)[x.numpy().tolist()]

def one_hot_for_y(y, class_count):
    y_one_hot = []
    for i in range(y.shape[0]):
        y_one_hot.append(one_hot(y[i], class_count).reshape(-1).numpy())
    return np.array(y_one_hot)

class RubiksCubeDataset(torch.utils.data.Dataset):
    def __init__(self, Xs, ys):
        self.Xs = Xs
        self.ys = ys

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index]

    def __len__(self):
        return len(self.Xs)

def collate_fn(batch):
    Xs, ys = zip(*batch)
    Xs_ori, Xs_tar = zip(*Xs)
    
    Xs_ori = np.array(Xs_ori)
    Xs_tar = np.array(Xs_tar)

    # ys = one_hot_for_y(torch.tensor(np.array(ys)), 24)
    ys = np.array(ys)
    return torch.tensor(Xs_ori, dtype=torch.long), torch.tensor(Xs_tar, dtype=torch.long), torch.tensor(ys, dtype=torch.float).unsqueeze(1)


class DynamicNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DynamicNetwork, self).__init__()
        self.seq_prepare_layers = nn.Sequential(
            nn.Embedding(24, 512),
            nn.Flatten(),
            nn.Linear(in_dim * 512, 2560),
            # There should be batch normalization here
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
            nn.Linear(1280, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )


    def forward(self, xs_ori, xs_tar):
        # Convert observation to tensor if it's a numpy array
        if isinstance(xs_ori, np.ndarray):
            xs_ori = torch.tensor(xs_ori, dtype=torch.float)
        
        if isinstance(xs_tar, np.ndarray):
            xs_tar = torch.tensor(xs_tar, dtype=torch.float)
        
        seq_prepare_out_ori = self.seq_prepare_layers(xs_ori)
        seq_prepare_out_tar = self.seq_prepare_layers(xs_tar)
        combined_input = torch.cat((seq_prepare_out_ori, seq_prepare_out_tar), dim=1)
        combined_recover_out = self.combined_recover_layers(combined_input)

        return combined_recover_out
    
    
