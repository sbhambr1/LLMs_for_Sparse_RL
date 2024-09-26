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
    
    reward_flags = [False, False, False]
    
    index_process_wood, index_plank, index_stick = -1, -1, -1
    
    for i, action in enumerate(plan):
        if action == '(get_processed_wood)':
            index_process_wood = i
        elif action == '(make_plank)':
            index_plank = i
        elif action == '(make_stick)':
            index_stick = i
    
    if index_process_wood != -1 and index_process_wood < index_plank and index_process_wood < index_stick:
        reward_flags[0] = True
        
    if index_stick != -1 and index_stick > index_process_wood:
        reward_flags[1] = True
        
    if index_plank != -1 and index_plank > index_process_wood:
        reward_flags[2] = True
    
    policy_save_file = f"{search_dir}{expt_name}.pkl"    
    
    with open(policy_save_file, 'wb') as f:
        pickle.dump(reward_flags, f)
        
    print(f"Reward flags saved to {policy_save_file}")
        