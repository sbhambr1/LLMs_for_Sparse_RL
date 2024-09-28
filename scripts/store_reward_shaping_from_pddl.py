import os
import cv2
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='Minecraft', help='Environment to use')
parser.add_argument('--variation', type=int, default=1, help='Variation of the LM policy')
parser.add_argument('--llm_model', type=str, default='gpt-4o', help='LLM model to use')
parser.add_argument('--llm_plan', type=str, default='llm_modulo', help='LLM plan to use, other option is "llm_modulo"')

if __name__ == '__main__':
    args = parser.parse_args()
        
    expt_name = 'reward_shaping_with_llm_plan'
    root_dir = os.getcwd()
    
    if args.llm_plan == 'vanilla':
        search_dir = f"{root_dir}/vanilla_llm_results/{args.llm_model}/{args.env}/pddl/variation_{args.variation}/"
    elif args.llm_plan == 'llm_modulo':
        search_dir = f"{root_dir}/llm_modulo_results/{args.llm_model}/{args.env}/pddl/variation_{args.variation}/"

    plan_file = search_dir + "llm_plan.txt"
    
    with open(plan_file, 'r') as f:
        plan = f.readlines()
    plan = [action.strip() for action in plan]
    
    reward_flags = [False, False, False, False, False]
    
    index_get_key0, index_door0, index_get_key1, index_door1, index_is_charged = -1, -1, -1, -1, -1
    
    for action in plan:
        if 'get_key0' in action:
            if index_get_key0 == -1:
                index_get_key0 = plan.index(action)
        if 'open_door0' in action:
            if index_door0 == -1:
                index_door0 = plan.index(action)
        if 'get_key1' in action:
            if index_get_key1 == -1:
                index_get_key1 = plan.index(action)
        if 'open_door1' in action:
            if index_door1 == -1:
                index_door1 = plan.index(action)
        if 'is_charged' in action:
            if index_is_charged == -1:
                index_is_charged = plan.index(action)
                
    if index_get_key0 != -1:
        reward_flags[0] = True
        
    if index_door0 != -1:
        if index_get_key0 != -1 and index_get_key0 < index_door0:
            reward_flags[1] = True
            
    if index_get_key1 != -1:
        if index_door0 != -1 and index_door0 < index_get_key1:
            reward_flags[2] = True
            
    if index_door1 != -1:
        if index_get_key1 != -1 and index_get_key1 < index_door1:
            reward_flags[3] = True
            
    if index_is_charged != -1:
        if index_door0 != -1 and index_door1 != -1 and index_door0 < index_is_charged and index_door1 < index_is_charged:
            reward_flags[4] = True
    
    policy_save_file = f"{search_dir}{expt_name}.pkl"    
    
    with open(policy_save_file, 'wb') as f:
        pickle.dump(reward_flags, f)
        
    print(f"Reward flags saved to {policy_save_file}")
        