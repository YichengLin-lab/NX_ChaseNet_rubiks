from ENV_WRAPER import EnvWraper
from UNIFIED_NX import Unified_NX
from NX_MODULE import NX_Module
from PPO_MODULE import PPO_Module
from ENV_NX_222 import RubiksCube222Env
from NETWORKS import *
from utils import *

if __name__ == "__main__":
    env = RubiksCube222Env()
    env_wraper = EnvWraper(env)
    nx_network = torch.load("./models_saved/nx_network_last.pt", map_location="cpu")
    actor_network = torch.load("./models_saved/ppo_actor_last.pt", map_location="cpu")
    nx_module = NX_Module(nx_network)

    test_results_list, mean_test_rews = test_model_recover_rate(actor_network, nx_module, test_batch_size=100)
    print("Recover rate: {}".format(test_results_list.count(1) / len(test_results_list)))
