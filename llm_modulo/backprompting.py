from llm_modulo.env_constraints import *

class LLM_Modulo:
    
    def __init__(self, env, seed):
        if env == 'MiniGrid-DoorKey-5x5-v0':
            self.env_critics = DoorKey5x5(env, seed)
        else:
            raise NotImplementedError
        
    def action_critic(self, llm_actions, llm_response, state):
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
        backprompt = ''
        if current_llm_action in feasible_actions[0]:
            return backprompt
        else:
            if current_llm_action == 'move forward':
                backprompt = "Information: You cannot 'move forward' in this state as you are facing a wall. Please choose another action."
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
            
        return backprompt
                
    
    
    
        
    