from llm_modulo.env_constraints import *
from utils.env_craft import Env_Craft
import random

class LLM_Modulo:
    
    def __init__(self, env_name, env):
        if env_name == "MineCraft-10x15":
            self.env = env
            self.env_critics = MineCraft10x15(env)
        else:
            raise NotImplementedError
        
    def action_critic(self, llm_actions, llm_response, state, give_feasible_actions=True):
        """
        Returns backprompt if actions is feasible in the environment in the current state.
        Input: llm_actions: actions taken by LLM (only actions that have been successfully executed in the environment so far) (list), 
                llm_response: action attempted by the llm in the current state (str),
                state (symbolic observation),
        Output: backprompt (str)
        """
        agent_pos = self.env_critics.get_agent_pos(state)
        current_llm_action = llm_response
        feasible_actions,message = self.env_critics.feasible_actions(agent_pos, llm_actions)
        random.shuffle(feasible_actions)  # Shuffle the list of feasible actions
        backprompt = ''
        FEASIBLE=False
    
        if current_llm_action in feasible_actions:
            FEASIBLE=True
            if 'extra' in message:
                backprompt = message['extra']
            return backprompt, FEASIBLE
        else:
            if current_llm_action == 'up':
                backprompt = "Information: You cannot take 'up' action in this state as you are facing a wall. Please choose another action."
            elif current_llm_action == 'down':
                backprompt = "Information: You cannot take 'down' action in this state as you are facing a wall. Please choose another action."
            elif current_llm_action == 'left':
                backprompt = "Information: You cannot take 'left' action in this state as you are facing a wall. Please choose another action."
            elif current_llm_action == 'right':
                backprompt = "Information: You cannot take 'right' action in this state as you are facing a wall. Please choose another action."
                            
        if give_feasible_actions:                            
            feasible = f'The following actions are feasible in this state: {feasible_actions}.'
            backprompt += feasible
            
        return backprompt, FEASIBLE