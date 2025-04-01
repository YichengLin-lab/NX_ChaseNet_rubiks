import numpy as np
import torch
from ENV_NX_222 import RubiksCube222Env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_action(ppo_actor, state):
    state = torch.tensor(state, dtype=torch.long).unsqueeze(0).to(DEVICE)
    action = ppo_actor(state)
    action_index = torch.argmax(action)
    
    return action_index.item()


def test_model_recover_rate(model, nx_module, test_batch_size=50):
    env = RubiksCube222Env()
    env.set_nx_ready(nx_module=nx_module)
    test_results_list = []
    test_mean_rewards = []
    
    for i in range(test_batch_size):

        state, _ = env.reset()
        done = False
        terminated = False
        truncated = False

        rews_list = []
        ep_actions = []

        while not done:
            action = get_action(model, state)
            ep_actions.append(action)
            state, reward, terminated, truncated, done, _ = env.step(action)
            rews_list.append(reward)
            
            if done:
                if terminated:
                    test_results_list.append(1)
                
                else:
                    test_results_list.append(0)
            
            
        test_mean_rewards.append(np.sum(rews_list))
        
        print("Test Round Episode Actions: {}".format(ep_actions))    
        
        
    return test_results_list, np.mean(test_mean_rewards)


def convert_state_list(state_list, device):
    """
    the original state_list is a list, with the element a tuple (original_state, target_state, n)
    we need to convert this to three lists: original_state_list, target_state_list, n_list
    so that it can be used to train the nx model
    """
    original_state_list = []
    target_state_list = []
    n_list = []

    for state_tuple in state_list:
        original_state, target_state, n = state_tuple
        original_state_list.append(original_state)
        target_state_list.append(target_state)
        n_list.append(n)
    

    # convert them to torch.tensor so that they can be used to train nx_network directly
    original_state_list = torch.tensor(np.array(original_state_list), dtype=torch.long)
    target_state_list = torch.tensor(np.array(target_state_list), dtype=torch.long)
    n_list = torch.tensor(n_list, dtype=torch.float32)
    
    return original_state_list.to(device), target_state_list.to(device), n_list.to(device)