from ENV_WRAPER import EnvWraper
from UNIFIED_NX import Unified_NX
from NX_MODULE import NX_Module
from PPO_MODULE import PPO_Module
from ENV_NX_222 import RubiksCube222Env
from NETWORKS import Nx_Network
from utils import *
import torch
from scipy.stats import spearmanr


if __name__ == "__main__":
    env = RubiksCube222Env()
    env_wraper = EnvWraper(env)

    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    nx_network = torch.load("./models_saved/nx_network_during_ppo_last.pt", map_location="cpu")

    nx_module = NX_Module(nx_network)
    ppo_module = PPO_Module(env)
    unified_nx = Unified_NX(env_wraper, nx_module, ppo_module)
    print("all_setup! Generating....")
    state_list = unified_nx.env_wraper.generate_nx()
    ori_state_tensor, tar_state_tensor, n_tensor = convert_state_list(state_list, nx_module.device)
    
    pre_out = unified_nx.nx_module.nx_network(ori_state_tensor, tar_state_tensor)
    pre_out = pre_out.squeeze(1)
    
    mse = torch.nn.MSELoss()
    loss = mse(pre_out, n_tensor)
    
    print("MSE Loss: {}".format(loss))
    
    # calculate the spearman correlation
    r = spearmanr(pre_out.detach().cpu().numpy().reshape(-1), n_tensor.detach().cpu().numpy().reshape(-1))
    
    print("Spearmanr: {}".format(r))


    
    
    
    
    
    
    
    
