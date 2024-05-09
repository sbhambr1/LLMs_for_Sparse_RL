import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper, StochasticActionWrapper
import argparse
import os
import sys
sys.path.insert(0,os.getcwd())
from utils.conversation import Conversation
from llm_modulo.backprompting import *
from llm_modulo.prompting import *
import warnings
warnings.filterwarnings("ignore")

# key_file = open(os.getcwd()+'/key.txt', 'r')
# API_KEY = key_file.readline().rstrip()

parser = argparse.ArgumentParser()

parser.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0", help="name of the environment to get LLM policy for")
parser.add_argument("--seed", type=int, default=0, help="environment seed to determine configuration")
parser.add_argument("--variation", type=int, default=0, help="Variation to prompt OpenAI's LLM (due to stochasticity at LLM's seed=0)")
parser.add_argument("--llm_model", default="gpt-3.5-turbo", help="LLM model to use for policy generation")
parser.add_argument("--add_text_desc", default=True, help="Whether to give additional text description of information when agent has picked up the key or opened the door")
# parser.add_argument("--give_feasible_actions", default=True, help="Whether to give feasible actions in backprompt")
# parser.add_argument("--give_tried_actions", default=True, help="Whether to give tried actions in backprompt")
parser.add_argument("--additional_expt_info", default="", help="Additional information for experiment")
parser.add_argument("--num_agent_steps", type=int, default=30, help="Number of steps the agent can take")
# parser.add_argument("--num_backprompt_steps", type=int, default=10, help="Number of backprompts that can be given")



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
    
    
    
        


def get_llm_policy(env, llm_model, conv, env_prompter, obs='', to_print=True, grid_text=False,give_add_text_desc=True,save_dir=None, num_agent_steps=30):
    if to_print:
        if grid_text:
            print(env_prompter.convert_obs_to_grid_text(obs))
        else:
            print(env_prompter.convert_obs_to_text(obs))

    llm_actions = []
    add_text_desc = ''

    for i in range(num_agent_steps):
        if llm_model != "None":
            prompt = env_prompter.get_step_prompt(obs, add_text_desc)
            print('-----------------')
            print('[STEP PROMPTING]---->')
            response = conv.llm_actor(prompt, stop=["\n"]).lower()
            action = [k for k, v in ACTION_DICT.items() if v in response][0]
            llm_actions.append(response)
        else:
            action = env.action_space.sample()
        print(prompt)
        print('LLM Response:', response)
        obs, reward, done, _, _ = env.step(action)
        # obs = convert_obs_to_text(obs)
        # prompt += f'Your action: {response}\n{obs}\n'
        
        if done:
            print('[LLM ACTIONS:] ---> ',llm_actions)
            
            policy_save_path = f'{save_dir}/llm_policy.txt'
            if not os.path.exists(os.path.dirname(policy_save_path)):
                os.makedirs(os.path.dirname(policy_save_path))
            with open(policy_save_path, 'w') as f:
                for action in llm_actions:
                    f.write("%s\n" % action)
            print('LLM policy has been saved to:', policy_save_path)
            return reward
    return 0

def main():

    args = parser.parse_args()
    
    if args.additional_expt_info == '':
        save_dir = f'./vanilla_llm_results/{args.llm_model}/{args.env}/step_wise/seed_{args.seed}/variation_{args.variation}'
    else:
        save_dir = f'./vanilla_llm_results/{args.llm_model}/{args.env}/step_wise/seed_{args.seed}/{args.additional_expt_info}/variation_{args.variation}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open(f'{save_dir}/log.txt', 'w')
    sys.stdout = log_file
    
    env = gym.make(args.env)
    env = SymbolicObsWrapper(env)
    conv = Conversation(args.llm_model)
    obs, _ = env.reset(seed=args.seed)
    
    env_prompter = MinigridPromptConstructor(args.env, seed=args.seed)

    total_reward = get_llm_policy(env=env, llm_model=args.llm_model, conv=conv, env_prompter=env_prompter, obs=obs, to_print=True, grid_text=True,give_add_text_desc=args.add_text_desc,save_dir=save_dir, num_agent_steps=args.num_agent_steps)
    print('-----------------')
    print(f"Total reward: {total_reward}")
    
if __name__ == "__main__":
    main()