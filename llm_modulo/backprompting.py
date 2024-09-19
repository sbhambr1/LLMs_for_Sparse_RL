from llm_modulo.env_constraints import *
from utils.env_mario import Env_Mario
import random

class LLM_Modulo:
    
    def __init__(self, env_name, env):
        if env_name == "Mario-8x11":
            self.env = env
            self.env_critics = Mario8x11(env)
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
        if agent_pos[0] == self.env.objects.door.location[0] and agent_pos[1] == self.envobjects.door.location[1]:
            if current_llm_action in feasible_actions:
                FEASIBLE=True
                if message["door"] == "both_keys":
                    backprompt = "Information: you are at door position but you can not open the door because you do not have both keys. Please find both keys first which are located down-stairs"
                elif message["door"] == "key":
                    backprompt = "Information: you are at door position and you have hidden key with you but you can not open the door because you do not have the other key. Please find the other key first which are located down-stairs"
                elif message["door"] == "hiidden_key":
                    backprompt = "Information: you are at door position and you have the key with you but you can not open the door because you do not have the hidden key. Please find the hidden key first which are located down-stairs"
                return backprompt, FEASIBLE
            else:
                if message["door"] == "both_keys":
                    backprompt = "Information: you are at door position but you can not open the door because you do not have both keys. Please find both keys first which are located down-stairs"
                elif message["door"] == "key":
                    backprompt = "Information: you are at door position and you have hidden key with you but you can not open the door because you do not have the other key. Please find the other key first which are located down-stairs"
                elif message["door"] == "hiidden_key":
                    backprompt = "Information: you are at door position and you have the key with you but you can not open the door because you do not have the hidden key. Please find the hidden key first which are located down-stairs"
                return backprompt, FEASIBLE
        elif current_llm_action in feasible_actions:
            FEASIBLE=True
            return backprompt, FEASIBLE
        else:
            if current_llm_action == 'up':
                if message["up"] == "wall":
                    backprompt = "Information: You cannot take 'up' action in this state as you are facing a wall. Please choose another action."
                elif message["up"] == "tube":
                    backprompt = "Information: You cannot take 'up' action in this state as you are in the tube. Please choose another action."
                elif message["up"] == "warn_ladder":
                    backprompt = "Information: You cannot take 'up' action in this state as the above ladder step is broken. Please choose another action."
                else:
                    ValueError("Bug in env_constraints.py file") 
            elif current_llm_action == 'down':
                if message["down"] == "wall":
                    backprompt = "Information: You cannot take 'down' action in this state as you are facing a wall. Please choose another action."
                elif message["up"] == "warn_ladder":
                    backprompt = "Information: You cannot take 'down' action in this state as the below ladder step is broken. Please choose another action."
                else:
                    ValueError("Bug in env_constraints.py file") 
            elif current_llm_action == 'left':
                backprompt = "Information: You cannot take 'left' action in this state as you are facing a wall. Please choose another action."
            elif current_llm_action == 'right':
                backprompt = "Information: You cannot take 'right' action in this state as you are facing a wall. Please choose another action."
                            
        if give_feasible_actions:                            
            feasible = f'The following actions are feasible in this state: {feasible_actions}.'
            backprompt += feasible
            
        return backprompt, FEASIBLE