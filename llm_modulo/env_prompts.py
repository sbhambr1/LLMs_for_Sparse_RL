import numpy as np

ACTION_DICT = {
    0: 'left', 
    1: 'right', 
    2: 'forward', 
    3: 'pickup key', 
    4: 'drop', 
    5: 'open door', 
    6: 'done'}
OBJECT_TO_IDX = {
    "unseen": -1,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}
DIRECTION_DICT = {
    'right': 0,
    'down': 1,
    'left': 2,
    'up': 3}

class EnvPrompts:
    
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
        
    def get_desc_obs(self, feasible_action):
        """
        Returns additional text description of information when agent has picked up the key or opened the door.
        Input: observation, feasible_action
        Output: additional text description
        """
        if feasible_action == 'turn left':
            return 'You have turned left.'
        elif feasible_action == 'turn right':
            return 'You have turned right.'
        elif feasible_action == 'move forward':
            return 'You have moved forward.' 
        elif feasible_action == 'pickup key':
            return 'You have picked up the key.'
        elif feasible_action == 'open door':
            return 'You have opened the door.'
        else:
            return ''
        
class DoorKey5x5Prompts(EnvPrompts):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        
    def get_step_prompt(self, obs, add_text_desc):
        TASK_DESC = "You are tasked with solving a 3x3 maze where you will encounter objects like a key and a door along with walls. Your task is 'use the key to open the door and then get to the goal'. You can be facing in any of the four directions. To move in any direction, to pick up the key, and to open the door, you need to face in the correct direction. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'turn left', 'turn right', 'move forward', 'pickup key', 'open door'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        if add_text_desc != '':
            step_prompt = f"{TASK_DESC}\n{add_text_desc}\n{OBS_DESC}\n{QUERY_DESC}\n"
        else:
            step_prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return step_prompt

    def get_prompt_with_backprompt(self, obs, backprompt, tried_actions, give_tried_actions=True):
        TASK_DESC = "You are tasked with solving a 3x3 maze where you will encounter objects like a key and a door along with walls. Your task is 'use the key to open the door and then get to the goal'. You can be facing in any of the four directions. To move in any direction, to pick up the key, and to open the door, you need to face in the correct direction. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'turn left', 'turn right', 'move forward', 'pickup key', 'open door'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        RETRY = "You have already tried the following actions: " + ', '.join(tried_actions) + ". Please choose another action."
        if give_tried_actions:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n{RETRY}\n"
        else:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return prompt_with_backprompt
    
    def _convert_pos_to_3x3(self, pos):
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
        obs_array = observation['image']
        agent_dir = observation['direction']
        agent_dir = list(DIRECTION_DICT.keys())[list(DIRECTION_DICT.values()).index(agent_dir)]
        col0, col1, col2 = [], [], []
        for i in range(3):
            # append key from OBJECT_TO_IDX    
            col0.append(obs_array[1][1:4][i][2])
            col1.append(obs_array[2][1:4][i][2])
            col2.append(obs_array[3][1:4][i][2])
            
        # transpose numpy array
        obs_matrix = np.array([col0, col1, col2])
        obs_matrix = np.transpose(obs_matrix)
        row0, row1, row2 = list(obs_matrix[0]), list(obs_matrix[1]), list(obs_matrix[2])
        
        for i in range(3):
            row0[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row0[i])]
            row1[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row1[i])]
            row2[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row2[i])]
                
        GRID_HEADER = "The current maze looks like this:\n"
        grid_text = f"{' '.join(row0)}\n{' '.join(row1)}\n{' '.join(row2)}\n"
        AGENT_DIR = f"You (agent) are currently facing {agent_dir}.\n"
        
        if 'key' not in row0 and 'key' not in row1 and 'key' not in row2:
            ADDITIONAL_INFO = "You have already picked up the key.\n"
            AGENT_DIR += ADDITIONAL_INFO
        
        text_obs = f"{GRID_HEADER}\n{grid_text}\n{AGENT_DIR}"
        return text_obs
        
class EmptyRandom5x5Prompts(EnvPrompts):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        
    def get_step_prompt(self, obs, add_text_desc):
        TASK_DESC = "You are tasked with solving a 3x3 maze where you will encounter walls. Your task is 'get to the goal'. You can be facing in any of the four directions. To move in any direction, you need to face in the correct direction. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'turn left', 'turn right', 'move forward'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        if add_text_desc != '':
            step_prompt = f"{TASK_DESC}\n{add_text_desc}\n{OBS_DESC}\n{QUERY_DESC}\n"
        else:
            step_prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return step_prompt

    def get_prompt_with_backprompt(self, obs, backprompt, tried_actions, give_tried_actions=True):
        TASK_DESC = "You are tasked with solving a 3x3 maze where you will encounter walls. Your task is 'get to the goal'. You can be facing in any of the four directions. To move in any direction, you need to face in the correct direction. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'turn left', 'turn right', 'move forward'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        RETRY = "You have already tried the following actions: " + ', '.join(tried_actions) + ". Please choose another action."
        if give_tried_actions:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n{RETRY}\n"
        else:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return prompt_with_backprompt
    
    def _convert_pos_to_3x3(self, pos):
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
        GOAL_DESC = f"The goal is at position {goal_pos}."
        MISC_DESC = f"There are walls at positions {walls}. There are unseen areas at positions {unseen}."
        OBS_DESC = f"The current observation is: \n{AGENT_DESC}\n{GOAL_DESC}\n{MISC_DESC}\n"
        return OBS_DESC

    def convert_obs_to_grid_text(self, observation):
        obs_array = observation['image']
        agent_dir = observation['direction']
        agent_dir = list(DIRECTION_DICT.keys())[list(DIRECTION_DICT.values()).index(agent_dir)]
        col0, col1, col2 = [], [], []
        for i in range(3):
            # append key from OBJECT_TO_IDX    
            col0.append(obs_array[1][1:4][i][2])
            col1.append(obs_array[2][1:4][i][2])
            col2.append(obs_array[3][1:4][i][2])
            
        # transpose numpy array
        obs_matrix = np.array([col0, col1, col2])
        obs_matrix = np.transpose(obs_matrix)
        row0, row1, row2 = list(obs_matrix[0]), list(obs_matrix[1]), list(obs_matrix[2])
        
        for i in range(3):
            row0[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row0[i])]
            row1[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row1[i])]
            row2[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row2[i])]
                
        GRID_HEADER = "The current maze looks like this:\n"
        grid_text = f"{' '.join(row0)}\n{' '.join(row1)}\n{' '.join(row2)}\n"
        AGENT_DIR = f"You (agent) are currently facing {agent_dir}.\n"
        
        text_obs = f"{GRID_HEADER}\n{grid_text}\n{AGENT_DIR}"
        return text_obs

class LavaGapS5Prompts(EnvPrompts):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        
    def get_step_prompt(self, obs, add_text_desc):
        TASK_DESC = "You are tasked with solving a 3x3 maze where you will encounter lava and walls. Your task is 'avoid the lava and get to the goal'. You can be facing in any of the four directions. To move in any direction, you need to face in the correct direction. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'turn left', 'turn right', 'move forward'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        if add_text_desc != '':
            step_prompt = f"{TASK_DESC}\n{add_text_desc}\n{OBS_DESC}\n{QUERY_DESC}\n"
        else:
            step_prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return step_prompt

    def get_prompt_with_backprompt(self, obs, backprompt, tried_actions, give_tried_actions=True):
        TASK_DESC = "You are tasked with solving a 3x3 maze where you will encounter lava and walls. Your task is 'avoid the lava and get to the goal'. You can be facing in any of the four directions. To move in any direction, you need to face in the correct direction. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'turn left', 'turn right', 'move forward'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        RETRY = "You have already tried the following actions: " + ', '.join(tried_actions) + ". Please choose another action."
        if give_tried_actions:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n{RETRY}\n"
        else:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return prompt_with_backprompt
    
    def _convert_pos_to_3x3(self, pos):
        return (pos[0]-1, pos[1]-1)

    def convert_obs_to_text(self, observation):
        # Convert observation to text
        # this has bugs due to the way the observation is structured (rows and cols are inverse in the observation array)
        obs_array = observation['image']
        
        walls = []
        unseen = []
        lavas = []
        for i in range(1,4):
            for j in range(1,4):
                if obs_array[i][j][2] == OBJECT_TO_IDX['agent']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    agent_pos = self._convert_pos_to_3x3((x, y))
                if obs_array[i][j][2] == OBJECT_TO_IDX['lava']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    lavas.append(self._convert_pos_to_3x3((x, y)))
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
        LAVA_DESC = f"There is lava at positions {lavas}."
        GOAL_DESC = f"The goal is at position {goal_pos}."
        MISC_DESC = f"There are walls at positions {walls}. There are unseen areas at positions {unseen}."
        OBS_DESC = f"The current observation is: \n{AGENT_DESC}\n{LAVA_DESC}\n{GOAL_DESC}\n{MISC_DESC}\n"
        return OBS_DESC

    def convert_obs_to_grid_text(self, observation):
        obs_array = observation['image']
        agent_dir = observation['direction']
        agent_dir = list(DIRECTION_DICT.keys())[list(DIRECTION_DICT.values()).index(agent_dir)]
        col0, col1, col2 = [], [], []
        for i in range(3):
            # append key from OBJECT_TO_IDX    
            col0.append(obs_array[1][1:4][i][2])
            col1.append(obs_array[2][1:4][i][2])
            col2.append(obs_array[3][1:4][i][2])
            
        # transpose numpy array
        obs_matrix = np.array([col0, col1, col2])
        obs_matrix = np.transpose(obs_matrix)
        row0, row1, row2 = list(obs_matrix[0]), list(obs_matrix[1]), list(obs_matrix[2])
        
        for i in range(3):
            row0[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row0[i])]
            row1[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row1[i])]
            row2[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row2[i])]
                
        GRID_HEADER = "The current maze looks like this:\n"
        grid_text = f"{' '.join(row0)}\n{' '.join(row1)}\n{' '.join(row2)}\n"
        AGENT_DIR = f"You (agent) are currently facing {agent_dir}.\n"
        
        text_obs = f"{GRID_HEADER}\n{grid_text}\n{AGENT_DIR}"
        return text_obs
     
class KeyCorridorS3R1Prompts(EnvPrompts):
    
    def __init__(self, env, seed):
        super().__init__(env, seed)
        
    def get_step_prompt(self, obs, add_text_desc):
        TASK_DESC = "You are tasked with solving a 1x5 maze where you will encounter objects like a key, an open door, a closed door, along with walls. Your task is 'use the key to open the closed door and then pick up the goal object'. You can be facing in any of the four directions. To move in any direction, you need to face in the correct direction. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'turn left', 'turn right', 'move forward', 'pickup key', 'pickup goal'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        if add_text_desc != '':
            step_prompt = f"{TASK_DESC}\n{add_text_desc}\n{OBS_DESC}\n{QUERY_DESC}\n"
        else:
            step_prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return step_prompt

    def get_prompt_with_backprompt(self, obs, backprompt, tried_actions, give_tried_actions=True):
        TASK_DESC = "You are tasked with solving a 1x5 maze where you will encounter objects like a key, an open door, a closed door, along with walls. Your task is 'use the key to open the closed door and then pick up the goal object'. You can be facing in any of the four directions. To move in any direction, you need to face in the correct direction. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'turn left', 'turn right', 'move forward', 'pickup key', 'pickup goal'.\n"
        OBS_DESC = self.convert_obs_to_grid_text(obs)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        RETRY = "You have already tried the following actions: " + ', '.join(tried_actions) + ". Please choose another action."
        if give_tried_actions:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n{RETRY}\n"
        else:
            prompt_with_backprompt = f"{TASK_DESC}\n{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return prompt_with_backprompt
    
    def _convert_pos_to_1x5(self, pos):
        return (pos[0]-1, pos[1]-1)

    def convert_obs_to_text(self, observation):
        # Convert observation to text
        # this has bugs due to the way the observation is structured (rows and cols are inverse in the observation array)
        obs_array = observation['image']
        
        walls = []
        unseen = []
        doors = []
        for i in range(1,2):
            for j in range(1,6):
                if obs_array[i][j][2] == OBJECT_TO_IDX['agent']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    agent_pos = self._convert_pos_to_1x5((x, y))
                if obs_array[i][j][2] == OBJECT_TO_IDX['door']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    doors.append(self._convert_pos_to_1x5((x, y)))
                if obs_array[i][j][2] == OBJECT_TO_IDX['key']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    key_pos = self._convert_pos_to_1x5((x, y))
                if obs_array[i][j][2] == OBJECT_TO_IDX['ball']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    ball_pos = self._convert_pos_to_1x5((x, y))
                if obs_array[i][j][2] == OBJECT_TO_IDX['wall']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    walls.append(self._convert_pos_to_1x5((x, y)))
                if obs_array[i][j][2] == OBJECT_TO_IDX['unseen']:
                    x, y = obs_array[i][j][0], obs_array[i][j][1]
                    unseen.append(self._convert_pos_to_1x5((x, y)))
        
        AGENT_DESC = f"In the 1x5 grid with (0, 0) at extreme left and (0,4) at extreme right, you are at position {agent_pos}."
        KEY_DESC = f"The key is at position {key_pos}."
        DOOR_DESC = f"The doors are at positions {doors}."
        GOAL_DESC = f"The goal object is at position {ball_pos}."
        MISC_DESC = f"There are walls at positions {walls}. There are unseen areas at positions {unseen}."
        OBS_DESC = f"The current observation is: \n{AGENT_DESC}\n{KEY_DESC}\n{DOOR_DESC}\n{GOAL_DESC}\n{MISC_DESC}\n"
        return OBS_DESC

    def convert_obs_to_grid_text(self, observation):
        
        # TODO: Fix this function by debugging the observation array
        obs_array = observation['image']
        agent_dir = observation['direction']
        agent_dir = list(DIRECTION_DICT.keys())[list(DIRECTION_DICT.values()).index(agent_dir)]
        
        row = []
        for i in range(1,6):
            row.append(obs_array[i][1][2])
            
        for i in range(5):
            row[i] = list(OBJECT_TO_IDX.keys())[list(OBJECT_TO_IDX.values()).index(row[i])]
            
        row[1] = 'open door'
        row[3] = 'closed door'
                
        GRID_HEADER = "The current maze looks like this:\n"
        grid_text = f"{' '.join(row)}"
        AGENT_DIR = f"You (agent) are currently facing {agent_dir}.\n"
        
        text_obs = f"{GRID_HEADER}\n[{grid_text}]\n{AGENT_DIR}"
        return text_obs
