import gymnasium as gym
# from llm_modulo_sparse_rl.scripts.llm_policy import get_initial_prompt
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
from utils.env_mario import Env_Mario

# key_file = open(os.getcwd()+'/key.txt', 'r')
# API_KEY = key_file.readline().rstrip()

parser = argparse.ArgumentParser()

parser.add_argument("--env_name", default="Mario-8x11", help="name of the environment to get LLM policy for")
parser.add_argument("--variation", type=int, default=0, help="Variation to prompt OpenAI's LLM (due to stochasticity at LLM's seed=0)")
parser.add_argument("--llm_model", default="gpt-3.5-turbo", help="LLM model to use for policy generation")
parser.add_argument("--add_text_desc", default=True, help="Whether to give additional text description of information when agent has picked up the key or opened the door")
parser.add_argument("--additional_expt_info", default="", help="Additional information for experiment")
    
    
def get_initial_prompt(env_name, obs, env_prompter, add_text_desc):
    if 'Mario' in env_name:
         TASK_DESC = "You are tasked with solving a 6x9 maze where you will encounter objects like a key, a hidden key, a ladder, a tude, and a door along with walls. Your task is 'first collect both the keys one of which is hidden in a red rock and then use them to open the door located at upstairs'. You can only go down through the tube and must use a worn-out ladder to go back up. The ladder will break after one use, and the door can only be opened after both keys are collected. You will be given a description of the maze at the first step and you need to choose the set of actions to take. The available actions are 'left', 'right', 'up', 'down'.\n"
        
    else:
        raise NotImplementedError

    OBS_DESC = env_prompter.convert_obs_to_grid_text(obs)

    QUERY_DESC = "What is the sequence of actions you will take to reach the goal? Output as a comma separated list. Do not include anything else in your response."
    if add_text_desc != '':
        step_prompt = f"{TASK_DESC}\n{add_text_desc}\n{OBS_DESC}\n{QUERY_DESC}\n"
    else:
        step_prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
    return step_prompt

def main():

    args = parser.parse_args()
    
    if args.additional_expt_info == '':
        save_dir = f'./vanilla_llm_results/{args.llm_model}/{args.env_name}/variation_{args.variation}'
    else:
        save_dir = f'./vanilla_llm_results/{args.llm_model}/{args.env_name}/variation_{args.variation}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open(f'{save_dir}/log.txt', 'w')
    sys.stdout = log_file
    
    if args.env_name == "Mario-8x11":
        env = Env_Mario(use_state=True, info_img=False)
    else:
        raise Exception("Environment not supported")
    
    conv = Conversation(args.llm_model)
    obs = env.reset()
    

    env_prompter = MarioPromptConstructor(args.env_name, env)
    initial_prompt = get_initial_prompt(args.env_name, obs, env_prompter=env_prompter, add_text_desc=args.add_text_desc)

    response = conv.llm_actor(initial_prompt, stop=["\n"]).lower()
    # action = [k for k, v in ACTION_DICT.items() if v in response][0]
    # llm_actions.append(response)

    print(initial_prompt)
    print('LLM Response:', response)

    print('-----------------')

    
if __name__ == "__main__":
    main()