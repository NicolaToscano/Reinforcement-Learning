# Libraries
import numpy as np
import random
from time import time
from Models import QNetwork 
from Utils import Utils
from Buffer import Buffer_DQN as Buffer
from Data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from gym_unity.envs import UnityEnv

device = "cpu"

class DQN():

    def __init__(self):

        # Hyperparameters
        self.learning_rate = 0.0003 
        self.buffer_size = 10240   
        self.batch_size = 1024           
        self.gamma = 0.99  
        self.update_every = 64
        self.max_steps = 100000
        
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.01                  

        self.summary_freq = 1000
        
        # Environment
        self.env_name = "Environments/env1/Unity Environment"
        channel = EngineConfigurationChannel()
        self.env = UnityEnv(self.env_name, worker_id=0, use_visual=False, side_channels=[channel], no_graphics = False, multiagent = False)
        channel.set_configuration_parameters(time_scale = 100)
        self.action_size, self.state_size  = Utils.getActionStateSize(self.env)
        self.n_agents = self.env.number_agents
        
        # Models
        self.local_model = QNetwork(self.state_size, self.action_size, seed = 0).to(device)
        self.target_model = QNetwork(self.state_size, self.action_size, seed = 0).to(device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.learning_rate)

        # Buffer memory
        self.memory = Buffer(self.buffer_size, self.batch_size, seed = 0, device = device)
        
        # Initialize time step (for updating every "update_every" time steps)
        self.t_step = 0

    def train(self):
       
        # Initial observation
        env_info = self.env.reset()
        state = env_info

        # Data
        self.data = Data(self.n_agents, self.summary_freq)
        
        # Training loop
        for _ in range(self.max_steps):

            # Action of agent 
            action, value = self.act(state)
              
            # Send the action to the environment
            next_state, reward, done, info = self.env.step(action) 
                    
            # Agent step
            self.step(state, action, reward, next_state, done)
                        
            # Update t_step
            self.t_step += 1
          
            # Next state
            state = next_state
                
            # Decrease epsilon
            if done:
                self.epsilon = max(self.epsilon_end, self.epsilon_decay*self.epsilon)
            
            # Update the score
            reward_ = np.expand_dims(reward, axis=0)
            value_ = value.unsqueeze(0)
            done_ = np.expand_dims(done, axis=0)
            self.data.update_score(reward_, value_, done_, self.t_step)

            # Summary
            if self.t_step % self.summary_freq == 0:
                self.data.summary(self.t_step)
        
        # Save
        self.save()
  
    def save(self):
        torch.save(self.local_model.state_dict(), 'Saved Models/model.pth')
        self.data.results()
        
    def load_model(self, model):
        self.model.load_state_dict(torch.load(model))
    
    def act(self, state):
      
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Get action values
        self.local_model.eval()
        with torch.no_grad():
            action_values = self.local_model(state)
        self.local_model.train()

        value = action_values.mean()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        
        return action, value

    def step(self, state, action, reward, next_state, done):
        
        # Save experience in buffer memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every "update_every" time steps
        if self.t_step % self.update_every == 0:
            
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def soft_update(self):
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(self.target_model.parameters(), self.local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def q_value(self,state,action):
        q_values = self.local_model(state)
        state_action_value = q_values.gather(1,action)
        return state_action_value
        
    def max_q_value(self,state):
        max_state_action_value = self.target_model(state).max(1)[0].detach()
        return max_state_action_value.unsqueeze(1)
    
    def learn(self, experiences):
        
        # Get Experiences
        states, actions, rewards, next_states, dones = experiences

        # Compute and minimize the loss
        state_action_values = self.q_value(states,actions)
        
        next_state_action_values = self.max_q_value(next_states)
        expected_state_action_values = (next_state_action_values * self.gamma*(1-dones)) + rewards
        
        # Loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)        
        
        # Optimizer step
        self.optimizerStep(self.optimizer, loss)        
        
        # Update target model
        self.soft_update()                     

    def optimizerStep(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()           
        optimizer.step()