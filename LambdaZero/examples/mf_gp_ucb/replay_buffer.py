import torch
import torch.utils.data                                                      
from collections import deque

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size        
        self.index = 0                      
        self.prev_transition = None                                      
                                                           
    def _add_to_buffer(self,transition):                
        if len(self.buffer) < self.max_size:         
            self.buffer.append(transition)               
        else:                                                            
            self.buffer[self.index] = transition           
            self.index = (self.index+1)%self.max_size                
                                 
    def add_transition(self, obs0, action0, reward, obs, terminal=False):
        transition = (obs0, action0, reward, obs, terminal)
        self._add_to_buffer(transition)                  
                                                                     
    def step(self, obs, reward, action, terminal=False):
        if self.prev_transition is not None:           
            obs0, reward0, action0 = self.prev_transition
            self.add_transition(obs0, action0, reward, obs, terminal)
        if terminal:                 
            self.prev_transition = None                
        else:                                          
            self.prev_transition = (obs, reward, action)
                                                       
    def __len__(self):              
        return len(self.buffer)                        
                                     
    def __getitem__(self, index):    
        return self.buffer[index]     
                                                       
    def state_dict(self):                              
        return {                                       
                'buffer': self.buffer,
                'index': self.index, 
                'prev_transition': self.prev_transition
        }                          
                                                       
    def load_state_dict(self, state):
        self.buffer = state['buffer']
        self.index = state['index']
        self.prev_transition = state['prev_transition']
