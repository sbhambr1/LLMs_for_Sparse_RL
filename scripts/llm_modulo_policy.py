import numpy as np
import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper, StochasticActionWrapper
from utils.conversation import Conversation
from llm_modulo.backprompting import *
import warnings
warnings.filterwarnings("ignore")

env_name = "MiniGrid-DoorKey-5x5-v0"
llm_model = "gpt-3.5-turbo" # "gpt-3.5-turbo", "None" for testing
SEED = 0
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
NUM_AGENT_STEPS = 20
STOCHASTIC = False

llm_modulo = LLM_Modulo(env_name, seed=0)

def get_initial_prompt(observation):
        TASK_DESC = "You are tasked with solving a 3x3 maze where you will encounter objects like a key and a door along with walls. Your task is 'use the key to open the door and then get to the goal'. You can be facing in any of the four directions. To move in any direction, to pick up the key, and to open the door, you need to face in the correct direction. You will be given a description of the maze at every step and you need to choose the next action to take. The available actions are 'turn left', 'turn right', 'move forward', 'pickup key', 'open door'.\n"
        
        OBS_DESC = convert_obs_to_grid_text(observation)
        
        QUERY_DESC = "What is the next action that you should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response.\n"
        initial_prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
        
        return initial_prompt
    
def get_step_prompt(obs):
    OBS_DESC = convert_obs_to_grid_text(obs)
    QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
    step_prompt = f"{OBS_DESC}\n{QUERY_DESC}\n"
    return step_prompt

def get_prompt_with_backprompt(obs, backprompt):
    OBS_DESC = convert_obs_to_grid_text(obs)
    QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
    prompt_with_backprompt = f"{backprompt}\n{OBS_DESC}\n{QUERY_DESC}\n"
    return prompt_with_backprompt

def _convert_pos_to_3x3(pos):
    return (pos[0]-1, pos[1]-1)

def convert_obs_to_text(observation):
    # Convert observation to text
    obs_array = observation['image']
    
    walls = []
    unseen = []
    for i in range(1,4):
        for j in range(1,4):
            if obs_array[i][j][2] == OBJECT_TO_IDX['agent']:
                x, y = obs_array[i][j][0], obs_array[i][j][1]
                agent_pos = _convert_pos_to_3x3((x, y))
            if obs_array[i][j][2] == OBJECT_TO_IDX['door']:
                x, y = obs_array[i][j][0], obs_array[i][j][1]
                door_pos = _convert_pos_to_3x3((x, y))
            if obs_array[i][j][2] == OBJECT_TO_IDX['key']:
                x, y = obs_array[i][j][0], obs_array[i][j][1]
                key_pos = _convert_pos_to_3x3((x, y))
            if obs_array[i][j][2] == OBJECT_TO_IDX['goal']:
                x, y = obs_array[i][j][0], obs_array[i][j][1]
                goal_pos = _convert_pos_to_3x3((x, y))
            if obs_array[i][j][2] == OBJECT_TO_IDX['wall']:
                x, y = obs_array[i][j][0], obs_array[i][j][1]
                walls.append(_convert_pos_to_3x3((x, y)))
            if obs_array[i][j][2] == OBJECT_TO_IDX['unseen']:
                x, y = obs_array[i][j][0], obs_array[i][j][1]
                unseen.append(_convert_pos_to_3x3((x, y)))
    
    AGENT_DESC = f"In the 3x3 grid with (0, 0) at top-left and (2,2) at bottom right, you are at position {agent_pos}."
    KEY_DESC = f"The key is at position {key_pos}."
    DOOR_DESC = f"The door is at position {door_pos}."
    GOAL_DESC = f"The goal is at position {goal_pos}."
    MISC_DESC = f"There are walls at positions {walls}. There are unseen areas at positions {unseen}."
    OBS_DESC = f"The current observation is: \n{AGENT_DESC}\n{KEY_DESC}\n{DOOR_DESC}\n{GOAL_DESC}\n{MISC_DESC}\n"
    return OBS_DESC

def convert_obs_to_grid_text(observation):
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
    
    # TODO: make obs more descriptive
    # Additional information if agent has picked up the key or opened the door
    if 'key' not in row0 and 'key' not in row1 and 'key' not in row2:
        ADDITIONAL_INFO = "You have already picked up the key.\n"
        AGENT_DIR += ADDITIONAL_INFO
    
    text_obs = f"{GRID_HEADER}\n{grid_text}\n{AGENT_DIR}"
    return text_obs


def get_llm_policy(env, conv, init_prompt, obs='', to_print=True, grid_text=False):
    if to_print:
        if grid_text:
            print(convert_obs_to_grid_text(obs))
        else:
            print(convert_obs_to_text(obs))
        
    llm_actions = []
    all_actions = []
    for i in range(NUM_AGENT_STEPS):
        if llm_model != "None":
            FEASIBLE = False
            for j in range(5):
                if i == 0 and j == 0:
                    prompt = init_prompt
                else:
                    if backprompt == '':
                        prompt = get_step_prompt(obs)
                    else:
                        prompt = get_prompt_with_backprompt(obs, backprompt)
                print(prompt)
                response = conv.llm_actor(prompt, stop=["\n"]).lower()
                backprompt = llm_modulo.action_critic(llm_actions=llm_actions, llm_response=response, state=obs)
                if backprompt == '':
                    FEASIBLE = True
                    llm_actions.append(response)
                all_actions.append(response)
                if FEASIBLE:
                    break
            if not FEASIBLE:
                print('LLM could not find a feasible action.')
                break
            llm_actions.append(response)
            action = [k for k, v in ACTION_DICT.items() if v in response][0]
        else:
            action = env.action_space.sample()
        
        print('LLM Response:', response)
        obs, reward, done, _, _ = env.step(action)
        if done:
            return reward
    return 0

def main():
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    env = SymbolicObsWrapper(env)
    conv = Conversation(llm_model)
    obs, _ = env.reset(seed=SEED)
    
    init_prompt = get_initial_prompt(obs)
    total_reward = get_llm_policy(env, conv, init_prompt, obs, to_print=False, grid_text=True)
    print('-----------------')
    print(f"Total reward: {total_reward}")
    
if __name__ == "__main__":
    main()