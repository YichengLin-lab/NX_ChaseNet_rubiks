from ENV_NX_222 import RubiksCube222Env
from NETWORKS import *
from utils import *

if __name__ == "__main__":
    env = RubiksCube222Env()
    actor_network = torch.load("./models_saved/ppo_actor_last.pt", map_location="cpu")
    
    done = False
    obs, _ = env.reset()
    actions_list = []

    while not done:
        action = get_action(actor_network, obs)
        actions_list.append(action)
        obs, reward, terminated, truncated, done, _ = env.step(action)
        env.render()
    
    print("Actions: {}".format(actions_list))
