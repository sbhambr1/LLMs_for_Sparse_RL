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
    
    reward_flags = [False, False, False, False]
    
    index_get_wood0, index_get_wood1, index_process_wood0, index_process_wood1, index_plank, index_stick = -1, -1, -1, -1, -1, -1
    
    w1, w2, m1, m2 = -1, -1, -1, -1
    
    """
    example plan: processwood1, make_plank, processwood2, make_stick, make_ladder
    
    find index_plank and index_stick if they exist
    find m = min(index_plank, index_stick)
    find index_process_wood0, index_process_wood1 if they exist
    find w = min(index_process_wood0, index_process_wood1)
    w < m => reward_flags[0] = True & also reward m
    w2 < m2 => reward_flags[1] = True & also reward m2
    """
    
    for action in plan:
        if 'get_wood' in action:
            if index_get_wood0 == -1:
                index_get_wood0 = plan.index(action)
            elif index_get_wood1 == -1:
                index_get_wood1 = plan.index(action)
        if 'get_processed_wood' in action:
            if index_process_wood0 == -1:
                index_process_wood0 = plan.index(action)
            elif index_process_wood1 == -1:
                index_process_wood1 = plan.index(action)
        if 'make_plank' in action:
            index_plank = plan.index(action)
        if 'make_stick' in action:
            index_stick = plan.index(action)
            
    if index_get_wood0 != -1 and index_process_wood0 != -1:
        if index_get_wood0 < index_process_wood0:
            reward_flags[0] = True
            
    if index_get_wood1 != -1 and index_process_wood1 != -1:
        if index_get_wood1 < index_process_wood1:
            reward_flags[1] = True
            
    # if both exist index_plank and index_stick
    if index_plank != -1 and index_stick != -1:
        if index_plank < index_stick:
            m1 = [index_plank, 'plank']
            m2 = [index_stick, 'stick']
        else:
            m1 = [index_stick, 'stick']
            m2 = [index_plank, 'plank']
        
        # if both exist index_process_wood0 and index_process_wood1
        if index_process_wood0 != -1 and index_process_wood1 != -1:
            w1 = min(index_process_wood0, index_process_wood1)
            w2 = max(index_process_wood0, index_process_wood1)
            if w1 < m1[0]:
                if m1[1] == 'stick':
                    reward_flags[2] = True
                elif m1[1] == 'plank':
                    reward_flags[3] = True
                    
            if w2 < m2[0]:
                if m2[1] == 'stick':
                    reward_flags[2] = True
                elif m2[1] == 'plank':
                    reward_flags[3] = True
                    
        elif index_process_wood0 != -1:
            w1 = index_process_wood0
            if w1 < m1[0]:
                if m1[1] == 'stick':
                    reward_flags[2] = True
                elif m1[1] == 'plank':
                    reward_flags[3] = True
                    
        elif index_process_wood1 != -1:
            w1 = index_process_wood1
            if w1 < m1[0]:
                if m1[1] == 'stick':
                    reward_flags[2] = True
                elif m1[1] == 'plank':
                    reward_flags[3] = True
                    
    elif index_stick != -1:
        # if both exist index_process_wood0 and index_process_wood1
        if index_process_wood0 != -1 and index_process_wood1 != -1:
            w1 = min(index_process_wood0, index_process_wood1)
            w2 = max(index_process_wood0, index_process_wood1)
                    
        elif index_process_wood0 != -1:
            w1 = index_process_wood0
                    
        elif index_process_wood1 != -1:
            w1 = index_process_wood1
            
        if w1 < index_stick:
                reward_flags[2] = True
                
    elif index_plank != -1:
        # if both exist index_process_wood0 and index_process_wood1
        if index_process_wood0 != -1 and index_process_wood1 != -1:
            w1 = min(index_process_wood0, index_process_wood1)
            w2 = max(index_process_wood0, index_process_wood1)
                    
        elif index_process_wood0 != -1:
            w1 = index_process_wood0
                    
        elif index_process_wood1 != -1:
            w1 = index_process_wood1
            
        if w1 < index_plank:
                reward_flags[3] = True
    
    
    policy_save_file = f"{search_dir}{expt_name}.pkl"    
    
    with open(policy_save_file, 'wb') as f:
        pickle.dump(reward_flags, f)
        
    print(f"Reward flags saved to {policy_save_file}")
        