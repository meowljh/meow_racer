from abc import *
import numpy as np
'''cf. ABC (Abstract Base Class)
- 흔히 코드의 "모듈화"를 진행하고자 할 때 base class의 super object로 ABC가 사용되는 것을 볼 수 있다.
- @abstractmethod: ABC로 정의된 class를 상속 받은 object는 이 abstractmethod가 없으면 initialize가 될 수 없다.
- @property: 함수로 정의가 되어 있지만 attribute로서 사용을 할 수 있다.
'''
class BaseBuffer(ABC):
    def __init__(self):
        self.first_store = True
    
    @abstractmethod
    def store(self, transitions):
        """store transitions to the buffer list"""
        
    @abstractmethod
    def sample(self, batch_size:int):
        """buffer에서 batch size만큼의 sample을 뽑음
        """
        transitions = [{}]
        return transitions
    
    def check_dim(self, transition):
        for key, val in transition.items():
            if len(val) > 1:
                for i in range(len(val)):
                    print(f"{key}{i}: {val[i].shape}")
            else:
                print(f"{key}: {val.shape}")
        self.first_store = False
        
    def stack_transitions(self, batch):

        transitions = {}
        # breakpoint()
        for key in batch[0].keys():
            if len(batch[0][key]) > 1:
                # breakpoint()
                b_list = []
                for i in range(len(batch[0][key])):
                    tmp_transition = np.stack([b[key][i][0] for b in batch], axis=0)
                    b_list.append(tmp_transition)
                transitions[key] = b_list
            else:
                transitions[key] = np.stack([b[key][0] for b in batch], axis=0) ## 여기서 쌓아주면 (#time, #dim)이 될 것임 
                ## batch: 배치 크기만큼 {key:value들} -> b: {key:value들} -> value는 (1, #dim) -> b[key][0] -> (#dim) 
    
        return transitions

class ReplayBuffer(BaseBuffer):
    '''memory buffer for transitions for off-policy RL methods'''
    def __init__(self, buffer_size):
        super(ReplayBuffer, self).__init__()
        self.buffer = np.zeros(buffer_size, dtype=dict) # 
        self.buffer_index = 0 # circular queue처럼 동작함 #
        self.buffer_size = buffer_size # 최대 replay memory buffer의 크기 #
        self.buffer_counter = 0 # size of the buffer #
    
    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])
        for transition in transitions: #all dictionary
            self.buffer[self.buffer_index] = transition
            self.buffer_index = (self.buffer_index+1)%self.buffer_size
            self.buffer_counter = min(self.buffer_counter+1, self.buffer_size)
    
    def sample(self, batch_size):
        batch_idx = np.random.randint(self.buffer_counter, size=batch_size) # batch size의 크기만큼 buffer에서 random sampling
        batch = self.buffer[batch_idx]
        transitions = self.stack_transitions(batch=batch)
        
        return transitions
    
    @property
    def size(self):
        return self.buffer_counter
    
class RolloutBuffer(BaseBuffer):
    '''memory buffer for transitions for on-policy RL methods'''
    def __init__(self):
        super(RolloutBuffer, self).__init__()
        self.buffer = []
    
    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])
        self.buffer += transitions
        
    def sample(self):
        transitions = self.stack_transitions(self.buffer)
        self.buffer = []
        
        return transitions

    @property
    def size(self):
        return len(self.buffer)