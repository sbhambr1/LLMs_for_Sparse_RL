from llm_modulo.env_constraints import *
import random

class LLM_Modulo:
    
    def __init__(self, env, seed):
        if env == 'MiniGrid-DoorKey-5x5-v0':
            self.env_critics = DoorKey5x5(env, seed)
        elif env == 'MiniGrid-Empty-Random-5x5-v0':
            self.env_critics = EmptyRandom5x5(env, seed)
        elif env == 'MiniGrid-LavaGapS5-v0':
            self.env_critics = LavaGapS5(env, seed)
        elif env == 'MiniGrid-KeyCorridorS3R1-v0':
            self.env_critics = KeyCorridorS3R1(env, seed)
        elif env == 'MiniGrid-DoorKey-6x6-v0':
            self.env_critics = DoorKey6x6(env, seed)
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
        agent_pos, agent_dir = self.env_critics.get_agent_pos(state)
        current_llm_action = llm_response
        feasible_actions = self.env_critics.feasible_actions(agent_pos, agent_dir, llm_actions)
        random.shuffle(feasible_actions)  # Shuffle the list of feasible actions
        backprompt = ''
        FEASIBLE=False
        if current_llm_action in feasible_actions:
            FEASIBLE=True
            return backprompt, FEASIBLE
        else:
            if current_llm_action == 'move forward':
                # backprompt = "Information: You cannot 'move forward' in this state as you are either facing a wall or an object. Please choose another action."
                backprompt = "Information: You cannot 'move forward' in this state as you are either facing a wall or an object. You can 'pickup key' if you are facing the key. You can 'open door' if you are facing the door. Please choose another action."

            elif current_llm_action == 'pickup key':
                if 'pickup key' in llm_actions:
                    backprompt = "Information: You have already picked up the key. Please choose another action."
                else:
                    backprompt = "Information: You cannot 'pickup key' in this state as you are not facing the key. Please choose another action."
            elif current_llm_action == 'open door':
                if 'open door' in llm_actions:
                    backprompt = "Information: You have already opened the door. Please choose another action."
                else:
                    backprompt = "Information: You cannot 'open door' in this state as you are not facing the door. Please choose another action."
                            
        if give_feasible_actions:                            
            feasible = f'The following actions are feasible in this state: {feasible_actions}.'
            backprompt += feasible
            
        return backprompt, FEASIBLE
                
    
    
    
        
    