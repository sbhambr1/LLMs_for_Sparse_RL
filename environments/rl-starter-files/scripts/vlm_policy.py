import os
from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper, StochasticActionWrapper
from utils.conversation import Conversation
import warnings
warnings.filterwarnings("ignore")


API_KEY=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")

env_name = "MiniGrid-DoorKey-5x5-v0"
llm_model = "gpt-4-vision-preview" # "gpt-4-vision-preview", "None" for testing
SEED = 0
ACTION_DICT = {
    0: 'turn left', 
    1: 'turn right', 
    2: 'move forward', 
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
image_save_dir = "./storage/DoorKey_VLM/"
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)
    
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

def get_prompt():
        TASK_DESC = "You are a decision making AI agent in a 3x3 grid world. You will encounter objects like key, door, and goal in the environment, along with walls. The task is 'use the key to open the door and then get to the goal'."
        OBS_DESC = "The following image represents the current state of the environment. The agent is represented by the red arrow, the key by the yellow key, the door by the yellow door, the goal by the green goal, walls by the grey blocks, and unseen areas by the black blocks. The agent is facing in the direction of the red arrow, and can only move in the direction it is facing. You have to be in an adjacent cell to the key facing it to pick it up, and in an adjacent cell as the door facing it to toggle/open it, and to be in the same cell as the goal to finish the task."
        QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. The available actions are 'turn left', 'turn right', 'move forward', 'pickup', 'drop', 'toggle', and 'done'. Note that 'turn left' and 'turn right' actions will turn the direction of the red arrow (agent). Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
        prompt = f"{TASK_DESC}\n{OBS_DESC}\n{QUERY_DESC}\n"
        return prompt
    
# def get_step_prompt():
#     OBS_DESC = "The following image represents the current state of the environment. The agent is represented by the red arrow, the key by the yellow key, the door by the yellow door, the goal by the green goal, walls by the grey blocks, and unseen areas by the black blocks. The agent is facing in the direction of the red arrow, and can only move in the direction it is facing. You have to be in an adjacent cell to the key facing it to pick it up, and in an adjacent cell as the door facing it to toggle/open it, and to be in the same cell as the goal to finish the task."
#     QUERY_DESC = "What is the next action that the agent should take? Only choose from the list of available actions. Do not include anything else in your response. For example, if you choose 'move forward', then only write 'move forward' in your response."
#     step_prompt = f"{OBS_DESC}\n{QUERY_DESC}\n"
#     return step_prompt

def get_vlm_policy(env, image_save_dir):
    obs, _ = env.reset(seed=SEED)
    for i in range(NUM_AGENT_STEPS):
        if llm_model != "None":
            prompt = get_prompt()
            img = env.render()
            cv2.imwrite(f"{image_save_dir}step_{i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_path = f"{image_save_dir}step_{i}.png"
            vlm_input_img = encode_image(image_path)
            response = get_vlm_response(prompt, vlm_input_img, llm_model).lower()
            print('VLM Response:', response)
            action = [k for k, v in ACTION_DICT.items() if v in response][0]
        else:
            action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        prompt += f'Your action: {response}\n{obs}\n'
        if done:
            return reward
    return 0

def main():
    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode='rgb_array')
    env = SymbolicObsWrapper(env)
    total_reward = get_vlm_policy(env, image_save_dir)
    print('-----------------')
    print(f"Total reward: {total_reward}")
    
if __name__ == "__main__":
    main()