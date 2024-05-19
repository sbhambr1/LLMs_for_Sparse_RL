import os
import cv2
import pickle
import argparse
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='MiniGrid-DoorKey-5x5-v0', help='Environment to use')
parser.add_argument('--seed', type=int, default=0, help='Seed for environment configuration')
parser.add_argument('--variation', type=int, default=1, help='Variation of the LM policy')

if __name__ == '__main__':
    args = parser.parse_args()


    search_dir = f"./llm_modulo_results/gpt-3.5-turbo/{args.env}/seed_{args.seed}/variation_{args.variation}/"
    policy_file = search_dir + "llm_policy.txt"
    
    policy = []
    with open(policy_file, 'r') as f:
        for line in f:
            action = line.strip()
            if action == 'pickup key':
                action = 'pickup'
            elif action == 'open door':
                action = 'toggle'
            policy.append(action)
            
    image_save_dir = policy_save_dir = f"./storage/lm_modulo_visualization/{args.env}/seed_{args.seed}/variation_{args.variation}/"
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
        
    policy_save_file = f"{policy_save_dir}lm_modulo_policy_ablation.pkl"

    ACTION_DICT = {
        0: 'turn left', 
        1: 'turn right', 
        2: 'move forward', 
        3: 'pickup', 
        4: 'drop', 
        5: 'toggle', 
        6: 'done'}
    ACTIONS_ENV = [list(ACTION_DICT.keys())[list(ACTION_DICT.values()).index(action)] for action in policy]

    env = gym.make(args.env, render_mode='rgb_array')

    store_policy = [] # list to store (state, action, shaped_reward) tuples for manual policy

    obs, _ = env.reset(seed=args.seed)
    done = False
    rewards = 0
    while not done:
        img = env.render()
        cv2.imwrite(f"{image_save_dir}step_{len(store_policy)}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        action = ACTIONS_ENV.pop(0)
        store_policy.append((obs, action, 0))
        obs, reward, done, _, info = env.step(action)
        rewards += reward
        if done:
            img = env.render()
            cv2.imwrite(f"{image_save_dir}step_{len(store_policy)}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print('[DONE]')
            print(f"Total rewards: {rewards}")
        
    env.close()

    # for each tuple in manual_policy, add the shaped reward by increasingly distributing total reward
    total_rewards = rewards
    n = len(store_policy)
    d = total_rewards/n
    for i in range(len(store_policy)):
        shaped_reward = d
        store_policy[i] = (store_policy[i][0], store_policy[i][1], shaped_reward)

    with open(policy_save_file, 'wb') as f:
        pickle.dump(store_policy, f)
    print(f"LM Modulo policy saved in {policy_save_file}!")
        