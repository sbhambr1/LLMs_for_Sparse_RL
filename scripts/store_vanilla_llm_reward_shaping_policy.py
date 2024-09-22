import os
import cv2
import pickle
import argparse
import gymnasium as gym
import warnings
import os
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='MiniGrid-DoorKey-5x5-v0', help='Environment to use')
parser.add_argument('--seed', type=int, default=0, help='Seed for environment configuration')
parser.add_argument('--variation', type=int, default=1, help='Variation of the LM policy')
parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo', help='LLM model to use')
parser.add_argument('--same_rewards_same_states', type=bool, default=True, help='Set this to True if you want to store the same rewards for the same states in the trajectory')
parser.add_argument('--query_type', type=str, default='entire_path', help='Query type for LLM: entire_path or step_wise')

if __name__ == '__main__':
    args = parser.parse_args()


    search_dir = f"./vanilla_llm_results/{args.llm_model}/{args.env}/{args.query_type}/seed_{args.seed}/variation_{args.variation}/"
    policy_file = search_dir + "llm_policy.txt"
    
    if os.path.exists(policy_file):
        print(f"[INFO] Reward shaping policy {policy_file} already exists at {policy_file}!")
    else:
        action_file = search_dir + "log.txt"
        
        llm_response = None
        with open(action_file, 'r') as f:
            for line in f:
                if line.startswith('LLM Response: '):
                    llm_response = line.strip()
                    
        llm_response = llm_response[14:].replace('.', '') # remove 'LLM Response: ' from the beginning and remove '.'

        # Split the llm_response by commas and remove any leading/trailing spaces
        actions = [action.strip() for action in llm_response.split(',')]
        
        # Create a new file called llm_policy.txt and write each action on a separate line
        with open(search_dir + "llm_policy.txt", "w") as f:
            for action in actions:
                f.write(action + "\n")
        
        if not os.path.exists(policy_file):
            print(f"Policy file {policy_file} not found!")
            exit(1)
        
        policy = []
        with open(policy_file, 'r') as f:
            for line in f:
                action = line.strip()
                if action == 'pickup key':
                    action = 'pickup'
                elif action == 'open door':
                    action = 'toggle'
                policy.append(action)
                
        image_save_dir = policy_save_dir = f"./storage/vanilla_llm_visualization/{args.llm_model}/{args.env}/{args.query_type}/seed_{args.seed}/variation_{args.variation}/"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
            
        policy_save_file = f"{policy_save_dir}vanilla_llm_policy.pkl"

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
            if len(ACTIONS_ENV) == 0:
                break
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
                print(f"[INFO] Total rewards: {rewards}")
                
        if not done:
            print("[INFO] Episode not done!")
            print(f"[INFO] Total rewards: {rewards}")
            
        env.close()

        if not args.same_rewards_same_states:
            # for each tuple in manual_policy, add the potentials
            if rewards == 0:
                total_rewards = 1 # since rewards=0 for not done episodes, so we assume 1
            else:
                total_rewards = rewards
            n = len(store_policy)
            d = ((2*total_rewards)/n)/(n-1)
            for i in range(len(store_policy)):
                potentials = (i+1)*d
                store_policy[i] = (store_policy[i][0], store_policy[i][1], potentials)
                
        else:
            if rewards == 0:
                total_rewards = 1 # since rewards=0 for not done episodes, so we assume 1
            else:
                total_rewards = rewards
            n = len(store_policy)
            d = ((2*total_rewards)/n)/(n-1)
            state_action_dict = {}
            for i in range(len(store_policy)):
                state = str(store_policy[i][0]['image'])
                action = store_policy[i][1]
                potentials = (i+1)*d
                if state in state_action_dict:
                    if action in state_action_dict[state]['actions']:
                        action_idx = state_action_dict[state]['actions'].index(action)
                        curr_potential = state_action_dict[state]['potentials'][action_idx]
                        potentials -= d
                else:
                    curr_potential = potentials
                    state_action_dict[state] = {'actions': [], 'potentials': []}
                    state_action_dict[state]['actions'].append(action)
                    state_action_dict[state]['potentials'].append(curr_potential)
                store_policy[i] = (store_policy[i][0], store_policy[i][1], curr_potential)   
            

        with open(policy_save_file, 'wb') as f:
            pickle.dump(store_policy, f)
        print(f"[INFO] Vanilla LLM policy saved in {policy_save_file}!")