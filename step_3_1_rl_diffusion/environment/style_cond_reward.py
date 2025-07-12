from reward.reward_shaping import *
from reward.penalty_shaping import *

class styleReward_Obj(object):
    def __init__(self,
                 style_config,
                 agent_config,
                 ):
        super().__init__()
        self.style_config = style_config
        self.agent_config = agent_config
    
    def _style_weight(self, style_setting:float):
        """
        [case1] NUM_STYLE_NET==2
        @style_setting ranges from 0 ~ 1, including 0, 0.5, 1
        if 0; defensive -> [1, 0, 1] -> Level 1
        elif 0.5; medium -> [1, 0.5, 0.5] -> Level 2
        else; aggressive -> [1, 1, 0] -> Level 3
        
        [case2] NUM_STYLE_NEW==3
        @style_setting ranges from 0 ~ 2, including 0, 1, 2
        if 0; defensive -> [1, 0, 1] -> Level 1
        elif 1; medium -> [1, 0.5, 0.5] -> Level 2
        else; aggressive -> [1, 1, 0] -> Level 3
        """
        num_style_net = self.agent_config['style']['num_nets']
        if (self.agent_config['style']['equal_weight_medium'] == False) and \
            ((num_style_net == 2 and style_setting == 0.5) or (num_style_net == 3 and style_setting == 1)): ##for MEDIUM style
                style_weight = np.array([
                    self.agent_config['style']['common_weight'], self.agent_config['style']['aggressive_weight'], self.agent_config['style']['defensive_weight']
                ])
            
        else:
            if num_style_net == 2:
                style_weight = np.array([1., style_setting, 1 - style_setting])
            elif num_style_net == 3:
                style_weight = np.array([1., style_setting / 2, 1 - (style_setting / 2)])
            else:
                raise UserWarning(f"{num_style_net} not supported for number of style networks")
        # return np.array([style_setting, 1 - style_setting])
        return style_weight

    def _calculate_step_reward(self,
                               style_type:str) -> float:
        current_reward = 0
        current_penalty = 0
        """CALCULATE THE STEP REWARD FOR THE STYLE SPECIFIES
        [TODO]
        """
        step_reward = current_reward + current_penalty
        return step_reward
    
    def _condition_w_style(self,
                           style_setting:float,
                           ) -> float:
        style_weight = self._style_weight(style_setting=style_setting)

        defensive_step_reward = self._calculate_step_reward(style_type="defensive")
        aggressive_step_reward = self._calculate_step_reward(style_type="aggressive")
        common_step_reward = self._calculate_step_reward(style_type="common")

        style_cond_step_reward = style_weight * np.array([common_step_reward, aggressive_step_reward, defensive_step_reward])
        style_cond_step_reward = style_cond_step_reward.sum()
        # style_cond_step_reward = style_weight * np.array([defensive_step_reward, aggressive_step_reward]).sum()
        # style_cond_step_reward += common_step_reward
        
        return style_cond_step_reward