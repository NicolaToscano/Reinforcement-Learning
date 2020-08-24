# Libraries
import numpy as np
import random
from time import time
from Models import ActorCritic 
from Utils import Utils
from Data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from gym_unity.envs import UnityEnv

device = "cpu"

class TEST():

    def __init__(self, n_episodes, env_name, model):
       
        # Nª Episodes
        self.n_episodes = n_episodes 

        # Environment
        self.env_name = env_name
        channel = EngineConfigurationChannel()
        self.env = UnityEnv(self.env_name, worker_id=0, use_visual=False, side_channels=[channel], no_graphics = False, multiagent = False)
        self.action_size, self.state_size  = Utils.getActionStateSize(self.env)
        
        # Model
        self.model = ActorCritic(self.state_size, self.action_size, seed = 0).to(device)
        
        # Initialize time step (for updating every "update_every" time steps)
        self.t_step = 1
        
        # Start test
        self.load_model(model)
        self.test()

    def test(self):
        
        # Initial observation
        env_info = self.env.reset()
        state = env_info

        # Data
        self.data = Data(1, 100)
        
        # Episodes done
        n_done = 0
        
        # Test loop
        while n_done <= self.n_episodes:

            # Action of agent 
            action, value = self.act(state)
              
            # Send the action to the environment
            next_state, reward, done, info = self.env.step(action) 
                        
            # Update t_step
            self.t_step += 1
            
            # Update n_done
            if done:
                n_done += 1
          
            # Next state
            state = next_state
                
            # Update the score
            reward_ = np.expand_dims(reward, axis=0)
            value_ = value.unsqueeze(0)
            done_ = np.expand_dims(done, axis=0)
            self.data.update_score(reward_, value_, done_, self.t_step)

            # Summary
            if done:
                self.data.summary(self.t_step)
   
    def load_model(self, model):
        self.model.load_state_dict(torch.load(model))
    
    def act(self, state):  
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Get actions probabilities and value from ActorCritic model
        self.model.eval()
        with torch.no_grad():
            action_probs, value = self.model(state)
        self.model.train()
            
        prob = F.softmax(action_probs, -1)
        
        # Get action and log of probabilities
        action = prob.multinomial(num_samples=1)
        
        return action, value