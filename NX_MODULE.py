import torch
from torch import nn
import numpy as np
import math
from utils import *

class NX_Module:
    """
    This is the main class of nx_module. 
    The nx_module contains a nx_network and a nx_optimizer. with some functions to train the nx_network.
    
    """
    def __init__(self, nx_network):
        self.nx_network = nx_network
        self.lr = 0.0001
        
        self.loss_fn = nn.MSELoss()
        self.training_epochs = 5
        self.loss_history = []
        # self.device = torch.device("mps" if torch.has_mps else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.nx_network = self.nx_network.to(self.device)
    
    def train(self, state_list_observed, during_ppo=False):
        """
        Train the nx_network with the state_list_observed.
        View the state_list_observed as a whole batch and train
        """
        
        if during_ppo:
            self.lr = 0.00001
        
        else:
            pass
        
        self.optimizer = torch.optim.Adam(self.nx_network.parameters(), lr=self.lr)

        self.nx_network.train()
        ori_state_tensor, tar_state_tensor, n_tensor = convert_state_list(state_list_observed, self.device)
        for epoch in range(self.training_epochs):

            self.optimizer.zero_grad()
            pre_out = self.nx_network(ori_state_tensor, tar_state_tensor)

            # before calculating loss, we need to confirm that pre_out and n_tensor have the same shape
            n_tensor = n_tensor.reshape(pre_out.shape)
            loss = self.loss_fn(pre_out, n_tensor)
            loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(loss.item())
            print("Epochs {}".format(epoch) + " Loss: {}".format(loss.item()))
            
    
    def get_reward(self, xs_ori, xs_tar):
        self.nx_network.eval()
        nx_output = self.nx_network(torch.tensor(xs_ori).unsqueeze(0).to(self.device), torch.tensor(xs_tar).unsqueeze(0).to(self.device)).item()
        if nx_output < 0:
            nx_output = 0
        try:
            reward = - math.log(nx_output, 1.2)
            
        except:
            # this could be caused by nx_ouptut == 0. So we add a tiny number to avoid that
            reward = - math.log(nx_output + 1e-4, 1.2)
            
        return reward
