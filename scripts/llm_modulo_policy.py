import numpy as np
import argparse
import os
import sys
sys.path.insert(0,os.getcwd())
import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper
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
parser.add_argument("--llm_model", default="gpt-3.5-turbo", help="LLM model to use for policy generation. Options include: gpt-3.5-turbo, gpt-4o-mini, gpt-4o, claude-3-haiku-20240307 (small), claude-3-sonnet-20240229 (medium), claude-3-opus-20240229 (large), meta.llama3-8b-instruct-v1:0")
parser.add_argument("--temp", type=float, default=0.0, help="temperature for LLM response")
parser.add_argument("--add_text_desc", default=True, help="Whether to give additional text description of information when agent has picked up the key or opened the door")
parser.add_argument("--give_feasible_actions", default=True, help="Whether to give feasible actions in backprompt")
parser.add_argument("--give_tried_actions", default=True, help="Whether to give tried actions in backprompt")
parser.add_argument("--additional_expt_info", default="", help="Additional information for experiment")
parser.add_argument("--num_agent_steps", type=int, default=30, help="Number of steps the agent can take")
parser.add_argument("--num_backprompt_steps", type=int, default=10, help="Number of backprompts that can be given")
    
def get_llm_policy(env, llm_model, llm_modulo, env_prompter, conv, obs='', to_print=True, grid_text=False, give_add_text_desc=True, give_feasible_actions=True, give_tried_actions=True, save_dir=None, num_agent_steps=30, num_backprompt_steps=10):
    if to_print:
        if grid_text:
            print(env_prompter.convert_obs_to_grid_text(obs))
        else:
            print(env_prompter.convert_obs_to_text(obs))
        
    llm_actions = []
    all_actions = []
    all_actions_formatted = []
    add_text_desc = ''
    for i in range(num_agent_steps):
        if llm_model != "None":
            FEASIBLE = False
            tried_actions = []
            for j in range(num_backprompt_steps):
                if len(tried_actions) == 0: # move to next step
                    prompt = env_prompter.get_step_prompt(obs, add_text_desc)
                    print('-----------------')
                    print('[STEP PROMPTING]---->')
                else:
                    prompt = env_prompter.get_prompt_with_backprompt(obs, backprompt, tried_actions, give_tried_actions=give_tried_actions)
                    print('------[BACK PROMPTING]---->')
                
                print(prompt)
                response = conv.llm_actor(prompt, stop=["\n"]).lower()
                backprompt, FEASIBLE = llm_modulo.action_critic(llm_actions=llm_actions, llm_response=response, state=obs, give_feasible_actions=give_feasible_actions)
                if FEASIBLE:
                    llm_actions.append(response)
                    if give_add_text_desc:
                        add_text_desc = env_prompter.get_desc_obs(response)
                    
                    break
                else:
                    tried_actions.append(response)
                all_actions_formatted.append(tried_actions)
                all_actions.append(response)
            if not FEASIBLE: # LLM could not find a feasible action in 10 (j) attempts
                print('LLM could not find a feasible action.')
                break
            action = [k for k, v in ACTION_DICT.items() if v in response][0]
        else:
            action = env.action_space.sample()
        
        print('LLM Response:', response)
        obs, reward, done, _, _ = env.step(action)
        if done:
            print('[LLM ACTIONS:] ---> ',llm_actions)
            
            policy_save_path = f'{save_dir}/llm_policy.txt'
            if not os.path.exists(os.path.dirname(policy_save_path)):
                os.makedirs(os.path.dirname(policy_save_path))
            with open(policy_save_path, 'w') as f:
                for action in llm_actions:
                    f.write("%s\n" % action)
            print('LLM policy has been saved to:', policy_save_path)
            
            all_tried_actions_save_path = f'{save_dir}/all_tried_actions.txt'
            if not os.path.exists(os.path.dirname(all_tried_actions_save_path)):
                os.makedirs(os.path.dirname(all_tried_actions_save_path))
            with open(all_tried_actions_save_path, 'w') as f:
                for actions in all_actions:
                    f.write("%s\n" % actions)
            print('All actions tried by the LLM have been saved to:', all_tried_actions_save_path)
            return reward
    return 0, llm_actions, all_actions

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    if args.additional_expt_info == '':
        save_dir = f'./llm_modulo_results/{args.llm_model}/{args.env}/seed_{args.seed}/variation_{args.variation}'
    else:
        save_dir = f'./llm_modulo_results/{args.llm_model}/{args.env}/seed_{args.seed}/{args.additional_expt_info}/variation_{args.variation}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open(f'{save_dir}/log.txt', 'w')
    sys.stdout = log_file
    
    env = gym.make(args.env)
    env = SymbolicObsWrapper(env)
    conv = Conversation(args.llm_model, args.temp)
    obs, _ = env.reset(seed=args.seed)
    llm_modulo = LLM_Modulo(args.env, seed=args.seed)
    env_prompter = MinigridPromptConstructor(args.env, seed=args.seed)
    returned_stuff = get_llm_policy(env=env, llm_model=args.llm_model, llm_modulo=llm_modulo, env_prompter=env_prompter, conv=conv, obs=obs, to_print=False, grid_text=True, give_add_text_desc=args.add_text_desc, give_feasible_actions=args.give_feasible_actions, give_tried_actions=args.give_tried_actions, save_dir=save_dir, num_agent_steps=args.num_agent_steps, num_backprompt_steps=args.num_backprompt_steps)
    if type(returned_stuff) == float or type(returned_stuff) == int:
        total_reward = returned_stuff
    else:
        # when it fails
        total_reward, llm_actions, all_actions = returned_stuff
        print('-----------------')
        print(f"LLM actions: {llm_actions}")
        print('-----------------')
        print(f"All actions tried by LLM: {all_actions}")
    # total_reward, llm_actions, all_actions = get_llm_policy(env=env, llm_model=args.llm_model, llm_modulo=llm_modulo, env_prompter=env_prompter, conv=conv, obs=obs, to_print=False, grid_text=True, give_add_text_desc=args.add_text_desc, give_feasible_actions=args.give_feasible_actions, give_tried_actions=args.give_tried_actions, save_dir=save_dir, num_agent_steps=args.num_agent_steps, num_backprompt_steps=args.num_backprompt_steps)

   
    
    print('-----------------')
    print(f"Total reward: {total_reward}")
    
    log_file.close()
    