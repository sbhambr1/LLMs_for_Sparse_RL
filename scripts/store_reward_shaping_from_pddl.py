import os
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='Mario-8x11', help='Environment to use')
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
    
    reward_flags = [False, False, False, False]
    
    index_visited_bottom, index_picked_key, index_picked_hidden_key, index_back_to_upper = -1, -1, -1, -1
    
    for i, action in enumerate(plan):
        if action == '(go_down_the_tube)':
            index_visited_bottom = i
        elif action == '(pickup_key)':
            index_picked_key = i
        elif action == '(pickup_hidden_key)':
            index_picked_hidden_key = i
        elif action == '(go_up_the_ladder)':
            index_back_to_upper = i
    
    if index_visited_bottom != -1:
        reward_flags[0] = True
            
    if index_picked_key != -1 and index_visited_bottom != -1 and index_picked_key > index_visited_bottom:
        reward_flags[1] = True
    
    if index_picked_hidden_key != -1 and index_visited_bottom != -1 and index_picked_hidden_key > index_visited_bottom:
        reward_flags[2] = True
        
    if index_back_to_upper != -1:
        if (index_picked_key != -1 and index_picked_key < index_back_to_upper) and (index_picked_hidden_key != -1 and index_picked_hidden_key < index_back_to_upper) and (index_visited_bottom != -1 and index_visited_bottom < index_back_to_upper):
            reward_flags[3] = True
    
    
    policy_save_file = f"{search_dir}{expt_name}.pkl"    
    
    with open(policy_save_file, 'wb') as f:
        pickle.dump(reward_flags, f)
        
    print(f"[INFO] Reward flags saved to {policy_save_file}")
        