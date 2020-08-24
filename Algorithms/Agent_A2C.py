# Libraries
import numpy as np
import random
from time import time
from Models import ActorCritic 
from Utils import Utils
from Buffer import Buffer_AC as Buffer
from Data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from gym_unity.envs import UnityEnv

device = "cpu"

class A2C():

    def __init__(self):
       
        # Hyperparameters
        self.learning_rate = 0.0003
        self.gamma = 0.99
        self.batch_size = 256
        self.max_steps = 100000 
        
        self.tau = 0.95
        self.entropy_coef = 0.001
        self.value_loss_coef = 0.5
        
        self.summary_freq = 1000
        
        self.gradients = []
        
        # Environment
        self.env_name = "Environments/env1/Unity Environment"
        channel = EngineConfigurationChannel()
        self.env = UnityEnv(self.env_name, worker_id=0, use_visual=False, side_channels=[channel], no_graphics = False, multiagent = True)
        channel.set_configuration_parameters(time_scale = 100)
        self.action_size, self.state_size  = Utils.getActionStateSize(self.env)
        self.n_agents = self.env.number_agents
        print("Nº of Agents: ",self.n_agents)
        
        # Model
        self.model = ActorCritic(self.state_size, self.action_size, seed = 0).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Buffer memory
        self.memory = []
        for _ in range(self.n_agents):
            self.memory.append(Buffer())
            
        # Initialize time step (for updating every "batch_size" time steps)
        self.t_step = 1

    def train(self):
        
        # Initial observation
        env_info = self.env.reset()
        state = env_info

        # Data
        self.data = Data(self.n_agents, self.summary_freq)
        
        # Training loop
        for _ in range(self.max_steps):
            
            action = []
            value = []

            # Action of agent
            for i in range(self.n_agents): 
                a,b = self.act(state[i])
                action.append(a)
                value.append(b)
              
            # Send the action to the environment
            next_state, reward, done, info = self.env.step(action) 
                    
            # Agent step
            for i in range(self.n_agents):
                self.step(state[i], action[i], reward[i], next_state[i], done[i], self.memory[i])
            
            # Next state
            state = next_state
                
            # Update the score
            self.data.update_score(reward, value, done, self.t_step)
           
            # Coordinator (update model)
            if self.t_step % self.batch_size == 0:
                self.coordinator()

             # Update t_step
            self.t_step += 1
            
            # Summary
            if self.t_step % self.summary_freq == 0:
                self.data.summary(self.t_step)
                    
        # Save
        self.save()
    
    def save(self):
        torch.save(self.model.state_dict(), 'Saved Models/model.pth')
        self.data.results()
        
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

    def step(self, state, action, reward, next_state, done, memory):
        
        # Save experience in buffer memory
        memory.add(state, action, reward, next_state, done)

        # Learn every "batch_size" time steps
        if self.t_step % self.batch_size == 0:
            experiences = memory.get()
            self.learn(experiences)
            memory.reset()
            
    def coordinator(self):
                
        # Loss
        loss = torch.stack(self.gradients)
        
        # Optimizer step
        self.optimizerStep(self.optimizer, loss.mean())         
        
        self.gradients = []
    
    def learn(self, experiences):
              
        # Get Experiences
        states, actions, rewards, next_states = experiences
        
        logits, values = self.model(states)
        probs     = F.softmax(logits, -1)
        log_probs = F.log_softmax(logits, -1)
        entropies = -(log_probs * probs).sum(1, keepdim=True)
        log_probs = log_probs.gather(1, actions.unsqueeze(1))
        
        _, value = self.model(next_states)
        values = torch.cat((values, value.data))
        
        policy_loss = 0
        value_loss = 0
        R = values[-1]
        gae = torch.zeros(1, 1)
        
        for i in reversed(range(len(rewards))):
            
            R = self.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + self.gamma * values[i + 1].data - values[i].data
            gae = gae * self.gamma * self.tau + delta_t
            policy_loss = policy_loss - (log_probs[i] * gae) - (self.entropy_coef * entropies[i])
           
        # Loss (gradient)
        loss = (policy_loss + self.value_loss_coef * value_loss)
        
        # Save gradient
        self.gradients.append(loss)
        
    def optimizerStep(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()           
        optimizer.step()