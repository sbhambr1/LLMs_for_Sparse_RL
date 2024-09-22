import numpy as np

OBJECT_TO_IDX = {
    "walkable_area": 0,
    "agent": 1,
    "wall": 2,
    "wood": 3,
    "workshop1": 4,
    "workshop2": 5,
    "workshop3": 6,
    "wood_process": 7,
}

IDX_TO_OBJECT = {
    0:"walkable_area",
    1:"agent",
    2:"wall",
    3:"wood",
    4:"workshop1",
    5:"workshop2",
    6:"workshop3",
    7:"wood_process",
}

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
        
class MineCarft10x15Prompts(EnvPrompts):
    
    def __init__(self, env):
        super().__init__(env)
        
    def get_step_prompt(self, obs, add_text_desc):
        TASK_DESC = "You are tasked with solving a 9x14 maze where you will encounter objects like raw wood blocks, a wood processing unit and 3 workshops with different functionaity. Workshop_1 make sticks from processed woods, workshop_2 make planks from processed wood and workshop_3 make ladder from sticks and planks.\
        Your task is 'Make a ladder from the row wood blocks. To do that first collect all raw wood blocks, bring them to wood processing unit to get processed woods, then make sticks and planks from the processed woods at workshop_1 and workshop_2 respectively. Then go to workshop_3 to make the ladder from the planks and sticks.'\
        You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'left', 'right', 'up', 'down'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'up', then only write 'up' in your response."
        if add_text_desc != '':
            step_prompt = f"{TASK_DESC}\n{add_text_desc}\n{OBS_DESC}\n{QUERY_DESC}\n"
        else:
            step_prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return step_prompt

    def get_prompt_with_backprompt(self, obs, backprompt, tried_actions, give_tried_actions=True):
        TASK_DESC = "You are tasked with solving a 9x14 maze where you will encounter objects like raw wood blocks, a wood processing unit and 3 workshops with different functionaity. Workshop_1 make sticks from processed woods, workshop_2 make planks from processed wood and workshop_3 make ladder from sticks and planks.\
        Your task is 'Make a ladder from the row wood blocks. To do that first collect all raw wood blocks, bring them to wood processing unit to get processed woods, then make sticks and planks from the processed woods at workshop_1 and workshop_2 respectively. Then go to workshop_3 to make the ladder from the planks and sticks.'\
        You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'left', 'right', 'up', 'down'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'up', then only write 'up' in your response."
        RETRY = "You have already tried the following actions: " + ', '.join(tried_actions) + ". Please choose another action."
        if give_tried_actions:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n{RETRY}\n"
        else:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return prompt_with_backprompt
    
    def _convert_pos_to_9x14(self, pos):
        return (pos[0]-1, pos[1]-1)

    def convert_obs_to_text(self, observation):
        # Convert observation to text
        # this has bugs due to the way the observation is structured (rows and cols are inverse in the observation array)
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
        MISC_DESC = f"There are walls at positions {walls}. There are unseen areas at positions {unseen}."
        OBS_DESC = f"The current observation is: \n{AGENT_DESC}\n{KEY_DESC}\n{DOOR_DESC}\n{GOAL_DESC}\n{MISC_DESC}\n"
        return OBS_DESC

    def convert_obs_to_grid_text(self, observation):
        obs_matrix = observation[0:150].reshape(10,15)[1:9,1:14]
        
        row0, row1, row2, row3, row4, row5, row6, row7 = list(obs_matrix[0]), list(obs_matrix[1]), list(obs_matrix[2]),\
                                             list(obs_matrix[3]), list(obs_matrix[4]),list(obs_matrix[5]),list(obs_matrix[6]),list(obs_matrix[7])
        
        for i in range(14):
            row0[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row0[i])]
            row1[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row1[i])]
            row2[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row2[i])]
            row3[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row3[i])]
            row4[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row4[i])]
            row5[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row5[i])]
            row6[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row6[i])]
            row7[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row7[i])]
                
        GRID_HEADER = "The current maze looks like this:\n"
        grid_text = f"{' '.join(row0)}\n{' '.join(row1)}\n{' '.join(row2)}\n{' '.join(row3)}\n{' '.join(row4)}\n{' '.join(row5)}\n{' '.join(row6)}\n{' '.join(row7)}\n"
        ADDITIONAL_INFO = ""

        if self.env.n_processed_wood == 2:
            ADDITIONAL_INFO += "You have collected all woods and they are processed.\n"
        if self.env.is_stick_made:
            ADDITIONAL_INFO += "You have made sticks from processed wood.\n"
        if self.env.is_plank_made:
            ADDITIONAL_INFO += "You have made planks from processed wood.\n"
           
        
        text_obs = f"{GRID_HEADER}\n{grid_text}\n{ADDITIONAL_INFO}"
        # text_obs = f"{GRID_HEADER}\n{grid_text}"
        return text_obs