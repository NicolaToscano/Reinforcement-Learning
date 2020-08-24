# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
import matplotlib.pyplot as plt

class Data():
    
    def __init__(self, n_agents, summary_freq):
        
        # Variables
        self.n_agents = n_agents
        self.summary_freq = summary_freq
        
        self.score = np.zeros(n_agents)
        self.score_print = []
        
        self.start_time = time()
        self.count = 0
        
        # Lists
        self.reward_temp = []
        self.value_temp = []
        self.reward_episode = []
        self.value_episode = []
        self.steps = []
        self.times = []  
            
    def time_elapsed(self):
        return int(time() - self.start_time)
    
    def update_score(self, reward, value, done, t_step):        
  
        for i in range(self.n_agents):
            self.score[i] += reward[i]
                
            if done[i]:
                self.score_print.append(self.score[i])
                self.data_save(self.score[i], value[i], False, 0, t_step)
                self.score[i] = 0
                    
        t = self.time_elapsed()
        self.data_save(0, 0, True, t, t_step)
    
    def data_save(self, reward, value, end_step, f_time, t_step):
        
        if (not end_step):
            self.reward_temp.append(reward)
            self.value_temp.append(value)
            self.count += 1
                
        if (end_step and t_step % self.summary_freq == 0):

                for i in range(self.count):
                    
                    self.reward_episode.append(self.reward_temp[i])
                    self.value_episode.append(self.value_temp[i])
                    self.steps.append(t_step)
                    self.times.append(int(f_time))
                    
                self.reward_temp = []
                self.value_temp = []
                self.count = 0
                
    def summary(self, t_step):
        print('Step: {}   Time Elapsed: {} s   Mean Reward: {:.3f}   Std of Reward: {:.3f}'
              .format(int(t_step), self.time_elapsed(), np.mean(self.score_print), np.std(self.score_print)))
        score_print = []
        
    def results(self):
        a = self.value_episode
        b = np.asarray(a)
        
        # Create DataFrame
        df = pd.DataFrame({'reward': self.reward_episode, 'value': b, 'step': self.steps, 'time (s)': self.times})
        
        # Plot
        sns.set(style="darkgrid")
        f, axes = plt.subplots(2, 2, figsize=(12, 12))
        s1 = sns.lineplot(data=df, x = "step", y = "reward", color="red", ax=axes[0, 0])
        s2 = sns.lineplot(data=df, x = "time (s)", y = "reward", color="red", ax=axes[1, 0])
        s3 = sns.lineplot(data=df, x = "step", y = "value", color="blue", ax=axes[0, 1])
        s4 = sns.lineplot(data=df, x = "time (s)", y = "value", color="blue", ax=axes[1, 1])
        
        # Save Plot
        f.savefig("Saved Plots/plot.png", dpi = 500)
        
        # Save Dataframe
        df.to_pickle("Saved DataFrames/dataframe.pkl") 