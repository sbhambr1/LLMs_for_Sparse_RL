import os
import cv2
import argparse
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Visualize environment')

parser.add_argument('--env', type=str, default='MiniGrid-DoorKey-5x5-v0', help='Environment name.')
parser.add_argument('--seed', type=int, default=0, help='Seed for determining the environment configuration.')

def visualize_env(env_name, seed):

    image_save_dir = f"./storage/env_visualization/{env_name}/seed_{seed}/"
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    env = gym.make(env_name, render_mode='rgb_array')

    obs, _ = env.reset(seed=seed)
    done = False
    rewards = 0
    img = env.render()
    cv2.imwrite(f"{image_save_dir}start_state.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
    env.close()

    print(f"Environment visualization saved in {image_save_dir}!")
    
if __name__ == '__main__':
    args = parser.parse_args()
    visualize_env(env_name=args.env, seed=args.seed)