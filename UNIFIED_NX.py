from copy import deepcopy
import torch


class Unified_NX:
    """
    This is the main class of the whole NX algorithm.
    
    
    """
    def __init__(self, env_wraper, nx_module, ppo_module) -> None:
        self.env_wraper = env_wraper 

        self.nx_module = nx_module
        self.pre_observe_time = 100

        self.ppo_module = ppo_module
        self.state_list_pre_observed = []
        
    
    def pre_observing(self):
        """
        Reset the environment and randomly take actions to observe the distance between states.
        Use this distance to train the nx_network.        
        
        """
        for i in range(self.pre_observe_time):
            print("NX Module PreObverving Round: {}".format(i))
            self.state_list_pre_observed = self.env_wraper.generate_nx()
            self.nx_module.train(self.state_list_pre_observed)
            torch.save(self.nx_module.nx_network, "./models_saved/nx_network_warmup.pt")
    
    def ppo_nx_simult_train(self):
        """
        Train the nx_network and ppo_network simultaneously.
        
        """
        
        self.env_wraper.env.set_nx_ready(self.nx_module)
        
        self.ppo_module.adopt_env(self.env_wraper.env)
        self.ppo_module.learn(1000)

        self.state_list_pre_observed = self.env_wraper.generate_nx(ppo_actor=self.ppo_module.actor)
        print(len(self.state_list_pre_observed))
        self.nx_module.train(self.state_list_pre_observed, during_ppo=True)


        
    
        
    


    
