from llm_modulo.env_prompts import *

class MarioPromptConstructor:
    
    def __init__(self, env_name, env):
        if env_name == "Mario-8x11":
            self.prompt_constructor = Mario8x11Prompts(env)
        else:
            raise NotImplementedError 
        
    def get_step_prompt(self, obs, add_text_desc):
        return self.prompt_constructor.get_step_prompt(obs, add_text_desc)
    
    def get_prompt_with_backprompt(self, obs, backprompt, tried_actions, give_tried_actions=True):
        return self.prompt_constructor.get_prompt_with_backprompt(obs, backprompt, tried_actions, give_tried_actions)

    def convert_obs_to_text(self, observation):
        return self.prompt_constructor.convert_obs_to_text(observation)
    
    def convert_obs_to_grid_text(self, observation):
        return self.prompt_constructor.convert_obs_to_grid_text(observation)

    def get_desc_obs(self, feasible_action):
        return self.prompt_constructor.get_desc_obs(feasible_action)