from llm_modulo.env_constraints import *

class LLM_Modulo:
    
    def __init__(self, env, seed):
        if env == 'MiniGrid-DoorKey-5x5-v0':
            self.env = DoorKey5x5(env, seed)
        else:
            raise NotImplementedError
        
    def action_critic(self, action, state):
        """
        Returns backprompt if actions is feasible in the environment in the current state.
        Input: action taken by LLM, state (symbolic observation), feasible actions from env_constraints
        Output: backprompt (str)
        """
        raise NotImplementedError
    
    
    
        
    