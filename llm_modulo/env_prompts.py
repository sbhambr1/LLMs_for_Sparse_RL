import numpy as np

OBJECT_TO_IDX = {
    "empty": 0,
    "agent": 1,
    "wall": 2,
    "tube": 3,
    "ladder": 4,
    "worn_ladder": 5,
    "hidden_key": 6,
    "key": 7,
    "door": 8,
    "dead_agent": 9}

IDX_TO_OBJECT = {
    0 : "empty",
    1 : "agent",
    2 : "wall",
    3 : "tube",
    4 : "ladder",
    5 : "worn_ladder",
    6 : "hidden_key",
    7 : "key",
    8 : "door",
    9 : "dead_agent"}

ACTION_DICT = {
    0: 'up', 
    1: 'down', 
    2: 'left', 
    3: 'right'}
    
TEXT_ACTION_DICT = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3}

class EnvPrompts:
    
    def __init__(self, env):
        self.env = env
        
    def get_desc_obs(self, feasible_action):
        """
        Returns additional text description of information when agent has picked up the key or opened the door.
        Input: observation, feasible_action
        Output: additional text description
        """
        if feasible_action == 'left':
            return 'You have moved to left cell.'
        elif feasible_action == 'right':
            return 'You have moved to right cell.'
        elif feasible_action == 'up':
            return 'You have moved to up cell.' 
        elif feasible_action == 'down':
            return 'You have moved to down cell.'
        else:
            return ''
        
class Mario8x11Prompts(EnvPrompts):
    
    def __init__(self, env):
        super().__init__(env)
        
    def get_step_prompt(self, obs, add_text_desc):
        TASK_DESC = "You are tasked with solving a 6x9 maze where you will encounter objects like a 'key', a 'hidden key', a 'ladder', a 'tube', and a 'door' along with 'walls'. Your task is to 'first collect both the keys located downstairs, and then use them to open the door located upstairs'. You can walk using the 'empty' cells. You can use the 'tube' to go down. You must use the 'ladder' to go back up. The 'door' can only be opened after both keys are collected. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'left', 'right', 'up', 'down'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'up', then only write 'up' in your response."
        if add_text_desc != '':
            step_prompt = f"{TASK_DESC}\n{add_text_desc}\n{OBS_DESC}\n{QUERY_DESC}\n"
        else:
            step_prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return step_prompt

    def get_prompt_with_backprompt(self, obs, backprompt, tried_actions, give_tried_actions=True):
        TASK_DESC = "You are tasked with solving a 6x9 maze where you will encounter objects like a 'key', a 'hidden key', a 'ladder', a 'tube', and a 'door' along with 'walls'. Your task is to 'first collect both the keys located downstairs, and then use them to open the door located upstairs'. You can walk using the 'empty' cells. You can use the 'tube' to go down. You must use the 'ladder' to go back up. The 'door' can only be opened after both keys are collected. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'left', 'right', 'up', 'down'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'up', then only write 'up' in your response."
        RETRY = "You have already tried the following actions: " + ', '.join(tried_actions) + ". Please choose another action."
        if give_tried_actions:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n{RETRY}\n"
        else:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return prompt_with_backprompt
    
    def _convert_pos_to_6x9(self, pos):
        return (pos[0]-1, pos[1]-1)

    def convert_obs_to_text(self, observation):
        # Convert observation to text
        obs_array = observation['image']
        
        walls = []
        unseen = []
        for i in range(1,4):
            for j in range(1,4):
                if obs_array[i][j][2] == OBJECT_TO_IDX['agent']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    agent_pos = self._convert_pos_to_3x3((x, y))
                if obs_array[i][j][2] == OBJECT_TO_IDX['door']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    door_pos = self._convert_pos_to_3x3((x, y))
                if obs_array[i][j][2] == OBJECT_TO_IDX['key']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    key_pos = self._convert_pos_to_3x3((x, y))
                if obs_array[i][j][2] == OBJECT_TO_IDX['goal']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    goal_pos = self._convert_pos_to_3x3((x, y))
                if obs_array[i][j][2] == OBJECT_TO_IDX['wall']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    walls.append(self._convert_pos_to_3x3((x, y)))
                if obs_array[i][j][2] == OBJECT_TO_IDX['unseen']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    unseen.append(self._convert_pos_to_3x3((x, y)))
        
        AGENT_DESC = f"In the 3x3 grid with (0, 0) at top-left and (2,2) at bottom right, you are at position {agent_pos}."
        KEY_DESC = f"The key is at position {key_pos}."
        DOOR_DESC = f"The door is at position {door_pos}."
        GOAL_DESC = f"The goal is at position {goal_pos}."
        MISC_DESC = f"There are walls at positions {walls}. There are empty areas at positions {unseen}."
        OBS_DESC = f"The current observation is: \n{AGENT_DESC}\n{KEY_DESC}\n{DOOR_DESC}\n{GOAL_DESC}\n{MISC_DESC}\n"
        return OBS_DESC

    def convert_obs_to_grid_text(self, observation):
        obs_matrix = observation.reshape(8,11)[1:7,1:10]
        
        row0, row1, row2, row3, row4, row5 = list(obs_matrix[0]), list(obs_matrix[1]), list(obs_matrix[2]),\
                                             list(obs_matrix[3]), list(obs_matrix[4]),list(obs_matrix[5])
        
        for i in range(9):
            row0[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row0[i])]
            row1[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row1[i])]
            row2[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row2[i])]
            row3[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row3[i])]
            row4[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row4[i])]
            row5[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row5[i])]
                
        GRID_HEADER = "The current maze looks like this:\n"
        grid_text = f"{' '.join(row0)}\n{' '.join(row1)}\n{' '.join(row2)}\n{' '.join(row3)}\n{' '.join(row4)}\n{' '.join(row5)}\n"
        
        if self.env.picked_hidden_key and self.env.picked_key:
            ADDITIONAL_INFO = "You have picked both keys. Try going to the door now.\n"
        elif self.env.picked_hidden_key:
            ADDITIONAL_INFO = "You have already picked up the hidden key. Try picking the other key now.\n"
        elif self.env.picked_key:
            ADDITIONAL_INFO = "You have already picked up the key. Try picking the hidden key now.\n"
        else:
            ADDITIONAL_INFO = ""
        
        text_obs = f"{GRID_HEADER}\n{grid_text}\n{ADDITIONAL_INFO}"
        # text_obs = f"{GRID_HEADER}\n{grid_text}"
        return text_obs