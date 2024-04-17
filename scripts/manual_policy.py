import os
import cv2
import pickle
import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper
import warnings
warnings.filterwarnings("ignore")

SEED=0
STOCHASTIC=False
ACTIONS_TO_TAKE = ['turn right', 'pickup', 'move forward', 'move forward', 'turn right', 'toggle', 'move forward', 'move forward', 'turn right', 'move forward', 'move forward', 'done']
ACTION_DICT = {
    0: 'turn left', 
    1: 'turn right', 
    2: 'move forward', 
    3: 'pickup', 
    4: 'drop', 
    5: 'toggle', 
    6: 'done'}
ACTIONS_ENV = [list(ACTION_DICT.keys())[list(ACTION_DICT.values()).index(action)] for action in ACTIONS_TO_TAKE]

image_save_dir = policy_save_dir = f"./storage/visualization/DoorKey_manual/seed_{SEED}/"
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)
    
policy_save_file = f"{policy_save_dir}manual_policy.pkl"

env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode='rgb_array')

manual_policy = [] # list to store (state, action, shaped_reward) tuples for manual policy

obs, _ = env.reset(seed=SEED)
done = False
rewards = 0
while not done:
    img = env.render()
    cv2.imwrite(f"{image_save_dir}step_{len(manual_policy)}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    action = ACTIONS_ENV.pop(0)
    manual_policy.append((obs, action, 0))
    obs, reward, done, _, info = env.step(action)
    rewards += reward
    if done:
        img = env.render()
        cv2.imwrite(f"{image_save_dir}step_{len(manual_policy)}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print('[DONE]')
        print(f"Total rewards: {rewards}")
    
env.close()

# for each tuple in manual_policy, add the shaped reward by increasingly distributing total reward
total_rewards = rewards
n = len(manual_policy)
d = ((2*total_rewards)/n)/(n-1)
for i in range(len(manual_policy)):
    shaped_reward = (i+1)*d
    manual_policy[i] = (manual_policy[i][0], manual_policy[i][1], shaped_reward)

with open(policy_save_file, 'wb') as f:
    pickle.dump(manual_policy, f)
print(f"Manual policy saved in {policy_save_file}!")
    