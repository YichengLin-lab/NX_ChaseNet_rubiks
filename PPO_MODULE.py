import torch
from torch import nn
import numpy as np
import math
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch
from torch.optim import Adam
from torch import nn
import numpy as np
from NETWORKS import PPO_Actor, PPO_Critic
from utils import *


class PPO_Module:
    def __init__(self, env):
        # Extract environment information
        self.env = env
        # self.device = torch.device("mps" if torch.has_mps else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.obs_dim = env.observation_space.shape[0]

        try:    
            self.act_dim = env.action_space.shape[0]
        except:
            self.act_dim = env.action_space.n

        # ALG STEP1
        # Initialize actor and critic networks
        self.actor = PPO_Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = PPO_Critic(self.obs_dim).to(self.device)

        self._init_hyperparameters()

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim, ), fill_value=0.5)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)


    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 6400
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 3
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.0001
        self.entropy_bonus_factor = 0.01

    def adopt_env(self, env):
        self.env = env  

    def get_action(self, obs):
        
        # Query the actor network for a mean action
        # Same thing as calling self.actor.forward(obs)
        sm = nn.Softmax(dim=1).to(self.device)
        obs = torch.tensor(obs).unsqueeze(0).to(self.device)
        
        probs = sm(self.actor(obs))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        # Create our Multivariate Normal Distribution
        # dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        # action = dist.sample()
        log_prob = dist.log_prob(action)
        

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid of 
        # the graph and just covert the action to numpy array. 
        # log prob as tensor is fine. Our computation graph will 
        # start later down the line.
        return action.cpu().detach().item(), log_prob.cpu().detach().numpy()        

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return
        # the shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            
            discounted_reward = 0 # the discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward) # this insert function is so fxxking good!

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs.to(self.device)
    
    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most
        # recent actor network
        # This segment of code is similar to that in get_action()
        sm = nn.Softmax(dim=1).to(self.device)
        probs = sm(self.actor(batch_obs))
        dist = Categorical(probs=probs)
        # dist = MultivariateNormal(mean, self.cov_mat)
        
        log_probs = dist.log_prob(batch_acts)

        # Return predicted values V and long pros log_probs
        return V.to(self.device), log_probs.to(self.device)

    def cal_entropy_bonus(self, batch_obs):
        sm = nn.Softmax(dim=1).to(self.device)
        policy_output = sm(self.actor(batch_obs))
        entropy = torch.distributions.Categorical(probs=policy_output).entropy()
        return entropy.mean()

    def rollout(self):
        # Batch data
        batch_obs = [] # batch observations
        batch_acts = [] # batch actions
        batch_log_probs = [] # log probs of each action
        batch_rews = [] # batch rewards
        batch_rtgs = [] # batch rewards-to-go
        batch_lens = [] # episodic lengths in batch
        
        # Number of timesteps run so far this batch
        t = 0


        while t < self.timesteps_per_batch:
            ep_rews = []
            obs_ori, _ = self.env.reset()
            obs = obs_ori
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # increment timesteps ran this batch so far
                t += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
    

                obs ,rew, _, _, done, someinfo = self.env.step(action)


                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
        
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)            

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)        
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).squeeze(1)
        
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs.to(self.device), batch_acts.to(self.device), batch_log_probs.to(self.device), batch_rtgs.to(self.device), batch_lens


    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far

        while t_so_far < total_timesteps: # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            
            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP 5
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # epoch code
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)


                # Calculate ratios

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                entropy_bonus = self.cal_entropy_bonus(batch_obs)

                # actor_loss = (-torch.min(surr1, surr2)).mean()
                actor_loss = (-torch.min(surr1, surr2)).mean() - (entropy_bonus * self.entropy_bonus_factor)

                # Calculate gradients and perform backward propagation for actor
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
















