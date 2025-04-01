import random
import torch 
from torch import nn


class EnvWraper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.reward_network = None
        self.max_n = 14
        self.nx_batch_size = 10240
    
    def get_action(self, net_work, obs, device):
        """
        net_work: policy network
        obs: observation
        device: cpu or cuda. Note that switching data from CPU to GPU at this process is very time consuming.
        """

        sm = nn.Softmax(dim=1)
        obs = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(device)

        probs = sm(net_work(obs))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        return action.item()
    
    def generate_nx(self, ppo_actor=None, device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Generate state_list with or without policy_actor
        if policy_actor == None, generate state_list with random action
        else, generate state_list with policy_actor

        """
        state_list = []
        if ppo_actor == None:

            while len(state_list) < self.nx_batch_size:
                ori_state, info = self.env.reset()
                state = ori_state

                last_last_move = None
                last_move = None
                
                for n in range(1, self.max_n + 1):
                    action = self.env.action_space.sample()
                    while action == last_move and action == last_last_move:
                        action = self.env.action_space.sample()

                    last_last_move = last_move
                    last_move = action                

                    state, _, _, _, done, info = self.env.step(action)
                    
                    if type(state) == type(None):
                        break

                    state_list.append((ori_state, state, n))

                    if done:
                        break

        else:
            self.nx_batch_size = 2560
            while len(state_list) < self.nx_batch_size:
                last_last_move = None
                last_move = None
                
                ori_state, info = self.env.reset()
                state = ori_state
                for n in range(1, self.max_n + 1):
                    action = self.get_action(ppo_actor, state, device)

                    if action == last_move and action == last_last_move:
                        break
                        
                    last_last_move = last_move
                    last_move = action

                    state, _, _, _, done, info = self.env.step(action)

                    if type(state) == type(None):
                        break

                    state_list.append((ori_state, state, n))
                    if done:
                        break
        
        random.shuffle(state_list)
        return state_list
    

if __name__ == "__main__":
    from ENV_NX_222 import RubiksCube222Env

    env = RubiksCube222Env()
    env_wraper = EnvWraper(env)
