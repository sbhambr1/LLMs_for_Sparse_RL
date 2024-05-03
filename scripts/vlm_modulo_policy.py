import numpy as np
import argparse
import os
import sys

sys.path.insert(0,os.getcwd())

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import requests
import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper
from utils.conversation import Conversation
from llm_modulo.backprompting import *
from llm_modulo.prompting import *
import warnings
warnings.filterwarnings("ignore")


key_file = open(os.getcwd()+'key.txt', 'r')
API_KEY = key_file.readline().rstrip()


parser = argparse.ArgumentParser()

parser.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0", help="name of the environment to get LLM policy for")
parser.add_argument("--seed", type=int, default=0, help="environment seed to determine configuration")
parser.add_argument("--variation", type=int, default=0, help="Variation to prompt OpenAI's LLM (due to stochasticity at LLM's seed=0)")
parser.add_argument("--vlm_model", default="gpt-4-vision-preview", help="VLM model to use for policy generation")
parser.add_argument("--add_text_desc", default=True, help="Whether to give additional text description of information when agent has picked up the key or opened the door")
parser.add_argument("--give_feasible_actions", default=True, help="Whether to give feasible actions in backprompt")
parser.add_argument("--give_tried_actions", default=True, help="Whether to give tried actions in backprompt")
parser.add_argument("--additional_expt_info", default="", help="Additional information for experiment")
parser.add_argument("--num_agent_steps", type=int, default=30, help="Number of steps the agent can take")
parser.add_argument("--num_backprompt_steps", type=int, default=10, help="Number of backprompts that can be given")
    
# def get_prompt():
#         TASK_DESC = "You are tasked with solving a 3x3 maze where you will encounter objects like a key and a door along with walls. Your task is 'use the key to open the door and then get to the goal'. You can be facing in any of the four directions. To move in any direction, to pick up the key, and to open the door, you need to face in the correct direction."
#         OBS_DESC = "The following image represents the current state of the environment. The agent is represented by the red arrow, the key by the yellow key, the door by the yellow door, the goal by the green goal, walls by the grey blocks, and unseen areas by the black blocks. The agent is facing in the direction of the red arrow, and can only move in the direction it is facing."
#         QUERY_DESC = "What is the next action that you should take? Only choose from the list of available actions. The available actions are: ['turn left', 'turn right', 'move forward', 'pickup key', 'open door']. Note that 'turn left' and 'turn right' actions will turn the direction of the red arrow (you). Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
#         prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
#         return prompt

     
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_vlm_response(prompt, img, model):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": model,
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                }
            }
            ]
        }
        ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']


def get_vlm_policy(env, vlm_model, llm_modulo, env_prompter, to_print=True, grid_text=False, give_add_text_desc=True, give_feasible_actions=True, give_tried_actions=True, save_dir=None, num_agent_steps=30, num_backprompt_steps=10):
    # init
    image_save_dir = f"./storage/visualization/DoorKey_VLM_seed_{args.seed}/"
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    obs, _ = env.reset(seed=args.seed)

    vlm_actions = []
    all_actions = []
    all_actions_formatted = []
    add_text_desc = ''

    # start the prompt loop

    for i in range(num_agent_steps): # max 30
        if vlm_model != "None":
            FEASIBLE = False
            tried_actions = []
            # img = env.render()
            for j in range(num_backprompt_steps): # max 10
                img = env.render()
                cv2.imwrite(f"{image_save_dir}step_{i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                image_path = f"{image_save_dir}step_{i}.png"
                vlm_input_img = encode_image(image_path)
                if len(tried_actions) == 0: # move to next step
                    prompt = env_prompter.get_step_prompt(obs, add_text_desc, vision=True) # obs is not used for vision, only img is used in the query
                    print('-----------------')
                    print('[STEP PROMPTING]---->')
                else:
                    prompt = env_prompter.get_prompt_with_backprompt(obs, backprompt, tried_actions, give_tried_actions=give_tried_actions, vision=True)
                    print('------[BACK PROMPTING]---->')

                print(prompt)
                response = get_vlm_response(prompt, vlm_input_img, vlm_model).lower()
                backprompt, FEASIBLE = llm_modulo.action_critic(llm_actions=vlm_actions, llm_response=response, state=obs, give_feasible_actions=give_feasible_actions)
                if FEASIBLE:
                    vlm_actions.append(response)
                    action = [k for k, v in ACTION_DICT.items() if v in response][0]
                    obs, reward, done, _, _ = env.step(action)
                    break
                else:
                    tried_actions.append(response)
                all_actions_formatted.append(tried_actions)
                all_actions.append(response)
            if not FEASIBLE: # VLM could not find a feasible action in 10 (j) attempts
                print('VLM could not find a feasible action.')
                break
        else:
            raise NotImplementedError

        # obs, reward, done, _, _ = env.step(action)
        if done:
            print('[VLM ACTIONS:] ---> ',vlm_actions)
            
            policy_save_path = f'{save_dir}/vlm_policy.txt'
            if not os.path.exists(os.path.dirname(policy_save_path)):
                os.makedirs(os.path.dirname(policy_save_path))
            with open(policy_save_path, 'w') as f:
                for action in vlm_actions:
                    f.write("%s\n" % action)
            print('VLM policy has been saved to:', policy_save_path)
            
            all_tried_actions_save_path = f'{save_dir}/all_tried_actions.txt'
            if not os.path.exists(os.path.dirname(all_tried_actions_save_path)):
                os.makedirs(os.path.dirname(all_tried_actions_save_path))
            with open(all_tried_actions_save_path, 'w') as f:
                for actions in all_actions:
                    f.write("%s\n" % actions)
            print('All actions tried by the VLM have been saved to:', all_tried_actions_save_path)
            return reward
    return 0



if __name__ == "__main__":
    
    args = parser.parse_args()
    
    if args.additional_expt_info == '':
        save_dir = f'./vlm_modulo_results/{args.env}/seed_{args.seed}/variation_{args.variation}'
    else:
        save_dir = f'./vlm_modulo_results/{args.env}/seed_{args.seed}/{args.additional_expt_info}/variation_{args.variation}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open(f'{save_dir}/log.txt', 'w')
    sys.stdout = log_file
    
    env = gym.make(args.env, render_mode='rgb_array')
    env = SymbolicObsWrapper(env)
    # conv = Conversation(args.llm_model)
    # obs, _ = env.reset(seed=args.seed)
    llm_modulo = LLM_Modulo(args.env, seed=args.seed)
    env_prompter = MinigridPromptConstructor(args.env, seed=args.seed)
    total_reward = get_vlm_policy(env=env, vlm_model=args.vlm_model, llm_modulo=llm_modulo, env_prompter=env_prompter, to_print=False, grid_text=True, give_add_text_desc=args.add_text_desc, give_feasible_actions=args.give_feasible_actions, give_tried_actions=args.give_tried_actions, save_dir=save_dir, num_agent_steps=args.num_agent_steps, num_backprompt_steps=args.num_backprompt_steps)
    
    print('-----------------')
    print(f"Total reward: {total_reward}")
    
    log_file.close()
    