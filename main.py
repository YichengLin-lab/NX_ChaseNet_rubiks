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

    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    nx_network = torch.load("./models_saved/nx_network_warmup.pt", map_location="cpu")
    
    nx_module = NX_Module(nx_network)
    ppo_module = PPO_Module(env)
    unified_nx = Unified_NX(env_wraper, nx_module, ppo_module)

    # print("Start Warming up!")

    # unified_nx.pre_observing()
    # print("Warm up complete!")
    # np.save("./nx_net_work_loss_pre_observing.npy", unified_nx.nx_module.loss_history)

    rews_list = []
    for i in range(100000):
        unified_nx.ppo_nx_simult_train()
        torch.save(ppo_module.actor, "./models_saved/ppo_actor_last.pt")
        test_results_list, mean_test_rews = test_model_recover_rate(ppo_module.actor, nx_module, test_batch_size=50)
        print("Recover rate: {}".format(test_results_list.count(1) / len(test_results_list)))
        print("Mean Reward Now: {}".format(mean_test_rews))
        
        nx_loss = unified_nx.nx_module.loss_history[-1]
        print("NX Loss Now: {}".format(nx_loss))
        torch.save(nx_module.nx_network, "./models_saved/nx_network_during_ppo_last.pt")
        
        rews_list.append(mean_test_rews)    
        np.save("./statistics/rews_list.npy", rews_list)














