from llm_modulo.env_prompts import *

class MinigridPromptConstructor:
    
    def __init__(self, env, seed):
        if env == 'MiniGrid-DoorKey-5x5-v0':
            self.prompt_constructor = DoorKey5x5Prompts(env, seed)
        elif env == 'MiniGrid-Empty-Random-5x5-v0':
            self.prompt_constructor = EmptyRandom5x5Prompts(env, seed)
        elif env == 'MiniGrid-LavaGapS5-v0':
            self.prompt_constructor = LavaGapS5Prompts(env, seed)
        elif env == 'MiniGrid-KeyCorridorS3R1-v0':
            self.prompt_constructor = KeyCorridorS3R1Prompts(env, seed)
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