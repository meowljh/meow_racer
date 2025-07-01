import os, sys
from torch.optim.lr_scheduler import LRScheduler

class BaseScheduler(LRScheduler):
    def __init__(self, optimizer, 
                 last_epoch=-1, 
    ):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        if last_epoch == -1:
            for group in optimizer.param_groups:
                initial_lr = group['lr']  
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param <initial_lr> not specified"
                    )
        
        self.base_lrs = [
            group['initial_lr'] for group in optimizer.param_groups
        ]


class StepDecay(BaseScheduler):
    def __init__(self, 
                 decay_ratio:float,
                 decay_steps:int,
                 optimizer, last_epoch=-1):
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)
        '''lr=lr_init * decay_ratio^(epoch // decay_steps)'''

class ExpDecay(BaseScheduler):
    def __init__(self,
                 decay_ratio:float,
                 optimizer, last_epoch=-1):
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)
        '''lr=lr_init * exp(-decay_rate * epoch)'''


class CosineDecay(BaseScheduler):
    def __init__(self,
                 min_lr:float,
                 max_lr:float,
                 T_max:int,
                 optimizer, last_epoch=-1):
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)
        '''lr=min_lr + (max_lr - min_lr) * 0.5 * (1_cos(pi * epoch / T_max))'''
        
        
class LinearDecay(BaseScheduler):
    def __init__(self,
                 final_lr:float,
                 optimizer, last_epoch=-1):
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)
        '''lr=lr_init - (lr_init-lr_final) * (epoch/total_epoch)'''
        

class RewardBased(BaseScheduler):
    def __init__(self,
                 mode:str, #choose from ['minimize', 'maximize'],
                 decay_ratio:float,
                 optimizer, last_epoch=-1):
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)
        '''Reduce LR on Plateau 
        decreases the learning rate when a certain metric value stops decreasing or increasing
        mode "minimize": when it stops decreasing
        mode "maximize": when it stops increasing
        '''
        
        
