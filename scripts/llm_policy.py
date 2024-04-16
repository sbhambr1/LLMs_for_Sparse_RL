import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper, StochasticActionWrapper
from llm_utils.conversation import Conversation
import warnings
warnings.filterwarnings("ignore")

env_name = "MiniGrid-DoorKey-5x5-v0"
llm_model = "gpt-3.5-turbo" # "gpt-3.5-turbo", "None" for testing
SEED = 0
ACTION_DICT = {
    0: 'left', 
    1: 'right', 
    2: 'forward', 
    3: 'pickup', 
    4: 'drop', 
    5: 'toggle', 
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
NUM_AGENT_STEPS = 20
STOCHASTIC = False


def get_initial_prompt(observation):
        TASK_DESC = "You are a decision making AI agent in a 3x3 grid world. You will encounter objects like key, door, and goal in the environment, along with walls. The task is 'use the key to open the door and then get to the goal'. You have to be in the same cell as the key to pick it up, and in the same cell as the door to open it. You have to be in the same cell as the goal to finish the task. The available actions are 'turn left', 'turn right', 'move forward', 'pickup', 'drop', 'toggle', and 'done'."
        OBS_DESC = convert_obs_to_text(observation)
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        initial_prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return initial_prompt
    
def get_step_prompt(observation_text):
    OBS_DESC = observation_text
    QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
    step_prompt = f"{OBS_DESC}\n{QUERY_DESC}\n"
    return step_prompt

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
    row0, row1, row2 = [], [], []
    for i in range(3):
        # append key from OBJECT_TO_IDX
        
        row0.append(obs_array[1][1:4][i][2])
        row1.append(obs_array[2][1:4][i][2])
        row2.append(obs_array[3][1:4][i][2])
    
    
    
        


def get_llm_policy(env, conv, init_prompt, obs='', to_print=True, grid_text=False):
    if to_print:
        if grid_text:
            print(convert_obs_to_grid_text(obs))
        else:
            print(convert_obs_to_text(obs))
    for i in range(NUM_AGENT_STEPS):
        if llm_model != "None":
            if i == 0:
                prompt = init_prompt
            else:
                prompt = get_step_prompt(obs)
            response = conv.llm_actor(prompt, stop=["\n"]).lower()
            action = [k for k, v in ACTION_DICT.items() if v in response][0]
        else:
            action = env.action_space.sample()
        print('LLM Response:', response)
        obs, reward, done, _, _ = env.step(action)
        obs = convert_obs_to_text(obs)
        prompt += f'Your action: {response}\n{obs}\n'
        if done:
            return reward
    return 0

def main():
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    env = SymbolicObsWrapper(env)
    conv = Conversation(llm_model)
    obs, _ = env.reset(seed=SEED)
    
    init_prompt = get_initial_prompt(obs)
    total_reward = get_llm_policy(env, conv, init_prompt, obs, to_print=True, grid_text=True)
    print('-----------------')
    print(f"Total reward: {total_reward}")
    
if __name__ == "__main__":
    main()