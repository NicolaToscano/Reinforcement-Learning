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

class A3C():

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
        
        # Environment
        self.env_name = "Environments/env1/Unity Environment"
        channel = EngineConfigurationChannel()
        self.env = UnityEnv(self.env_name, worker_id=0, use_visual=False, side_channels=[channel], no_graphics = False, multiagent = True)
        channel.set_configuration_parameters(time_scale = 100)
        self.action_size, self.state_size  = Utils.getActionStateSize(self.env)
        self.n_agents = self.env.number_agents
        print("Nº of Agents: ",self.n_agents)
        
        # Shared model
        self.shared_model = ActorCritic(self.state_size, self.action_size, seed = 0).to(device)
        
        # Agents models
        self.agent_model = []
        self.optimizer = []
        for i in range(self.n_agents):
            self.agent_model.append(ActorCritic(self.state_size, self.action_size, seed = 0).to(device))
            self.optimizer.append(optim.Adam(self.agent_model[i].parameters(), lr=self.learning_rate))
        
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
        
        # Variables
        self.n_episode = np.ones(self.n_agents)
        
        # Training loop
        for _ in range(self.max_steps):
            
            action = []
            value = []

            # Action of agent
            for i in range(self.n_agents): 
                a,b = self.act(state[i], self.agent_model[i])
                action.append(a)
                value.append(b)
              
            # Send the action to the environment
            next_state, reward, done, info = self.env.step(action) 
                    
            # Agent step
            for i in range(self.n_agents):
                self.step(state[i], action[i], reward[i], next_state[i], done[i], self.agent_model[i], self.memory[i], self.optimizer[i])
                        
            # Update t_step
            self.t_step += 1
          
            # Next state
            state = next_state
                
            # Update the score
            self.data.update_score(reward, value, done, self.t_step)
           
            # Update shared_model
            self.update_shared_model(done)

            # Summary
            if self.t_step % self.summary_freq == 0:
                self.data.summary(self.t_step)
        
        # Save
        self.save()
  
    def save(self):
        torch.save(self.shared_model.state_dict(), 'Saved Models/model.pth')
        self.data.results()
        
    def load_model(self, model):
        for i in range(self.n_agents):
            self.agent_model[i].load_state_dict(torch.load(model))
    
    def act(self, state, agent_model):  
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Get actions probabilities and value from ActorCritic model
        agent_model.eval()
        with torch.no_grad():
            action_probs, value = agent_model(state)
        agent_model.train()
            
        prob = F.softmax(action_probs, -1)
        
        # Get action and log of probabilities
        action = prob.multinomial(num_samples=1)
        
        return action, value

    def step(self, state, action, reward, next_state, done, agent_model, memory, optimizer):
        
        # Save experience in buffer memory
        memory.add(state, action, reward, next_state, done)
        
        # Learn every "batch_size" time steps
        if self.t_step % self.batch_size == 0:
            experiences = memory.get()
            self.learn(experiences, agent_model, optimizer)
            memory.reset()
            
    def shared_to_agent(self, agent_model):
        agent_model.load_state_dict(self.shared_model.state_dict())
    
    def agent_to_shared(self, agent_model):
        self.shared_model.load_state_dict(agent_model.state_dict())
    
    def update_shared_model(self, done):
        
        for i in range(self.n_agents):
            
                if done[i]:
                    
                    # shared model -> agent model
                    if self.n_episode[i] % 5 == 0:
                        self.shared_to_agent(self.agent_model[i])
                    
                    # agent model -> shared model 
                    else:
                        self.agent_to_shared(self.agent_model[i])
                    
                    self.n_episode[i] += 1
    
    def learn(self, experiences, agent_model, optimizer):
              
        # Get Experiences
        states, actions, rewards, next_states = experiences
        
        logits, values = agent_model(states)
        probs     = F.softmax(logits, -1)
        log_probs = F.log_softmax(logits, -1)
        entropies = -(log_probs * probs).sum(1, keepdim=True)
        log_probs = log_probs.gather(1, actions.unsqueeze(1))
        
        _, value = agent_model(next_states)
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
            
        # Loss
        loss = (policy_loss + self.value_loss_coef * value_loss)
        
        # Optimizer step
        self.optimizerStep(optimizer, loss)
    
    def optimizerStep(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()           
        optimizer.step()