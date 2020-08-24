# Libraries
import numpy as np
import random
from time import time
from Models import ActorCritic 
from Utils import Utils
from Buffer import Buffer_PPO as Buffer
from Data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from gym_unity.envs import UnityEnv

device = "cpu"

class PPO():

    def __init__(self):
              
        # Hyperparameters
        self.learning_rate = 0.0003            
        self.betas = (0.9, 0.999)        
        self.gamma = 0.99               
        self.eps_clip = 0.2              
        self.buffer_size = 2048
        self.batch_size = 256        
        self.K_epochs = 3
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
        
        # Model
        self.model = ActorCritic(self.state_size, self.action_size, seed = 0).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas = self.betas)   
        self.MseLoss = nn.MSELoss()
        
        # Buffer memory
        self.memory = []
        for _ in range(self.n_agents):
            self.memory.append(Buffer())
        
        # Initialize time step (for updating when buffer_size is full)
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
            logprobs = []
            value = []
            
            # Action of agent
            for i in range(self.n_agents): 
                a,b,c = self.act(state[i])
                action.append(a)
                logprobs.append(b)
                value.append(c)
                 
            # Send the action to the environment
            next_state, reward, done, info = self.env.step(action) 
                    
            # Done
            done_ = []
            for i in range(self.n_agents):
                done_.append(1-done[i])
            
            # Agent step
            for i in range(self.n_agents):
                self.step(state[i], action[i], reward[i], next_state[i], done_[i], logprobs[i], value[i], self.memory[i])
                       
            # Update t_step
            self.t_step += 1

            # Next state
            state = next_state
                
            # Update the score
            self.data.update_score(reward, value, done, self.t_step)
            
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
        log_probs = F.log_softmax(action_probs, -1)
        
        # Get action and log of probabilities
        action = prob.multinomial(num_samples=1)
        log_probs = log_probs.gather(1, action)
        
        return action, log_probs, value

    def step(self, state, action, reward, next_state, done, logprobs, value, memory):

        # Update model when buffer_size is full
        if memory.len_() == (self.buffer_size/self.n_agents):
            self.learn()
            for i in range(self.n_agents):
                self.memory[i].reset()
                
        # Save experience in buffer memory
        memory.add(state, action, reward, next_state, done, logprobs, value)
    
    def evaluate(self, states, next_states, actions, rewards, masks, compute_gae):
        
        logits, values = self.model(states)
        probs     = F.softmax(logits, -1)
        log_probs = F.log_softmax(logits, -1)
        entropies = -(log_probs * probs).sum(1, keepdim=True)
        log_probs = log_probs.gather(1, actions.unsqueeze(1))
        
        values_ = values
        
        _, value = self.model(next_states)
        values = torch.cat((values, value.data))
        
        returns = []
       
        if(compute_gae):

            gae = torch.zeros(1, 1)
            
            for i in reversed(range(len(rewards))):

                # Generalized Advantage Estimation
                delta_t = rewards[i] + self.gamma * masks[i] * values[i + 1].data - values[i].data
                gae = gae * self.gamma * self.tau * masks[i] + delta_t
            
                returns.insert(0, gae + values[i])   
        
        return log_probs, values_, entropies, returns
    
    def compute_returns(self):
        
        returns_= []
        
        for i in range(self.n_agents):
            
            # Get Experiences (of each agent)
            experiences = self.memory[i].get()
            states, actions, rewards, next_states, dones, logprobs_, values_ = experiences
            
            # Evaluate
            _, _, _, r = self.evaluate(states, next_states, actions, rewards, dones, compute_gae = True)
            returns_.append(r)

        l = []

        for i in range(len(returns_)):
            for j in range(len(returns_[0])):
                l.append(returns_[i][j])
        
        return l    
    
    def learn(self):

        # Get Experiences
        states, actions, rewards, next_states, dones, logprobs_, values_ = self.getExp()
   
        returns_eval = self.compute_returns()
        returns_eval = torch.tensor(returns_eval).to(device)
        returns_eval = returns_eval.unsqueeze(1)
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            
            # List with all indices
            l = np.arange(self.buffer_size)
            l = list(l)
            
            x = self.buffer_size // self.batch_size
            
            for _ in range(x):
            
                # Take a random batch
                indices = random.sample(l, self.batch_size)
            
                old_logprobs = torch.empty(self.batch_size, 1)
                old_values = torch.empty(self.batch_size, 1)
                old_actions = torch.empty(self.batch_size)
                old_states = torch.empty(self.batch_size, self.state_size)
                old_next_states = torch.empty(self.batch_size, self.state_size)
                old_rewards = np.zeros(self.batch_size)
                returns = torch.empty(self.batch_size, 1)

                for i in range(len(indices)):
                    
                    old_logprobs[i] = logprobs_[indices[i]]
                    old_values[i] = values_[indices[i]]
                    old_actions[i] = actions[indices[i]]
                    old_states[i] = states[indices[i]]
                    old_next_states[i] = next_states[indices[i]]
                    old_rewards[i] = rewards[indices[i]]
                    returns[i] = returns_eval[indices[i]]
            
                old_actions = old_actions.long()
                
                # Remove indices to not repeat
                for i in indices:
                    l.remove(i)
            
                # Evaluate
                logprobs, state_values, dist_entropy, _ = self.evaluate(old_states, old_next_states, old_actions,
                                                                        rewards, dones, compute_gae = False)
            
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs)
         
                # Finding Surrogate Loss:
                advantages = returns - old_values
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
                # LOSS = ACTOR LOSS + CRITIC_DISCOUNT * CRITIC_LOSS - ENTROPY_BETA * ENTROPY
                loss = -torch.min(surr1, surr2)+self.value_loss_coef*self.MseLoss(state_values, returns)-self.entropy_coef*dist_entropy

                # Optimizer step
                self.optimizerStep(self.optimizer, loss.mean()) 
  
    def optimizerStep(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()           
        optimizer.step()
    
    def getExp(self):
        
        states, actions, rewards, next_states, dones, logprobs, values = [], [], [], [], [], [], []
        
        for i in range(self.n_agents):
            experiences = self.memory[i].get()
            states.append(experiences[0])
            actions.append(experiences[1])
            rewards.append(experiences[2])
            next_states.append(experiences[3])
            dones.append(experiences[4])
            logprobs.append(experiences[5])
            values.append(experiences[6])
        
        states_, actions_, rewards_, next_states_, dones_, logprobs_, values_ = [], [], [], [], [], [], []
        
        for i in range(len(states)):
            for j in range(len(states[0])):
                states_.append(states[i][j])
                actions_.append(actions[i][j])
                rewards_.append(rewards[i][j])
                next_states_.append(next_states[i][j])
                dones_.append(dones[i][j])
                logprobs_.append(logprobs[i][j])
                values_.append(values[i][j])
        
        states__ = torch.empty(self.buffer_size, self.state_size)
        actions__ = torch.empty(self.buffer_size)
        next_states__ = torch.empty(self.buffer_size, self.state_size)
        dones__ = torch.empty(self.buffer_size)
        logprobs__ = torch.empty(self.buffer_size, 1, 1)
        values__ = torch.empty(self.buffer_size)

        for i in range(self.buffer_size):
            states__[i] = states_[i]
            actions__[i] = actions_[i]
            next_states__[i] = next_states_[i]
            dones__[i] = dones_[i]
            logprobs__[i] = logprobs_[i]
            values__[i] = values_[i]   
            
        return states__, actions__, rewards_, next_states__, dones__, logprobs__, values__               