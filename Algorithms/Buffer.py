# Libraries
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim

class Buffer_DQN:

    def __init__(self, buffer_size, batch_size, seed, device):
     
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Buffer_AC:

    def __init__(self):
         self.experiences = []

    def add(self, state, action, reward, next_state, done):
        self.experiences.append((state.tolist(), action, reward, next_state.tolist()))
        
    def reset(self):
        self.experiences = []

    def get(self):
        exp =  np.array(self.experiences)
        states  = torch.tensor(exp[:, 0].tolist())
        actions = torch.tensor(exp[:, 1].tolist())
        rewards = torch.tensor(exp[:, 2].tolist())
        next_states = torch.tensor(exp[:, 3].tolist())
        return (states, actions, rewards, next_states)

class Buffer_PPO:

    def __init__(self):
         self.experiences = []

    def add(self, state, action, reward, next_state, done, logprobs, value):
        self.experiences.append((state.tolist(), action, reward, next_state.tolist(), done, logprobs.tolist(), value))

    def reset(self):
        self.experiences = []

    def get(self):
        exp =  np.array(self.experiences)
        states  = torch.tensor(exp[:, 0].tolist())
        actions = torch.tensor(exp[:, 1].tolist())
        rewards = (exp[:, 2].tolist())
        next_states = torch.tensor(exp[:, 3].tolist())
        dones = torch.tensor(exp[:, 4].tolist())
        logprobs = torch.tensor(exp[:, 5].tolist())
        values = torch.tensor(exp[:, 6].tolist())
        return (states, actions, rewards, next_states, dones, logprobs, values)

    def len_(self):
        return len(self.experiences)