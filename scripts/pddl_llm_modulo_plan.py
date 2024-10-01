import os
import sys
import argparse
sys.path.insert(0,os.getcwd())
from utils.conversation import Conversation

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--llm_model", type=str, default="gpt-4o", help="Model to use for LLM")
parser.add_argument("--variation", type=int, default=1, help="Variation of the LLM query if seed != 0.")
parser.add_argument("--num_agent_steps", type=int, default=10, help="Number of steps the agent can take")
parser.add_argument("--num_backprompt_steps", type=int, default=5, help="Number of backprompts that can be given")


#SOLUTION bfws-ff plan: (go_down_the_tube), (pickup_hidden_key), (pickup_key), (go_up_the_ladder), (unlock_door)
    
def get_initial_prompt(pddl_domain, pddl_problem, llm_plan_so_far):
    
    TASK_DESC = "Here is a pddl domain, a planning problem. Provide only the next action for the query problem. Provide only the pddl syntax for the plan where the action is represented as (ACTION_NAME OBJECTS). Do not provide anything else in your response.\n\n"
    
    DOMAIN_DESC = f"[DOMAIN]\n{pddl_domain}\n\n"
    PROBLEM_DESC = f"[PROBLEM]\n{pddl_problem}\n\n"    
    QUERY_DESC = "[YOUR RESPONSE]\n"
    
    if llm_plan_so_far == []:
        step_prompt = TASK_DESC + DOMAIN_DESC + PROBLEM_DESC + QUERY_DESC
    else:    
        step_prompt = TASK_DESC + DOMAIN_DESC + PROBLEM_DESC + 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + QUERY_DESC
    return step_prompt

def get_planning_constraints(llm_response, llm_plan_so_far):
    
    FEASIBLE = False
    backprompt = ''
    
    valid_actions = ['(go_down_the_tube)', '(pickup_hidden_key)', '(pickup_key)', '(go_up_the_ladder)', '(unlock_door)']
    
    if llm_response not in valid_actions:
        FEASIBLE = False
        backprompt = 'Your response: ' + llm_response + '\n\n' + 'The action provided is not a valid action. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
    else:
        if llm_response == '(go_down_the_tube)':
            if '(go_up_the_ladder)' in llm_plan_so_far or llm_plan_so_far == []:
                FEASIBLE = True
            else:
                FEASIBLE = False
                backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                
        elif llm_response == '(pickup_hidden_key)':
            if '(go_down_the_tube)' in llm_plan_so_far:
                FEASIBLE = True
            else:
                FEASIBLE = False
                backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                
        elif llm_response == '(pickup_key)':
            if '(go_down_the_tube)' in llm_plan_so_far:
                FEASIBLE = True
            else:
                FEASIBLE = False
                backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                
        elif llm_response == '(go_up_the_ladder)':
            if '(pickup_hidden_key)' in llm_plan_so_far and '(pickup_key)' in llm_plan_so_far:
                FEASIBLE = True
            else:
                FEASIBLE = False
                backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                
        elif llm_response == '(unlock_door)':
            if '(go_up_the_ladder)' in llm_plan_so_far:
                FEASIBLE = True
            else:
                FEASIBLE = False
                backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
    
    return FEASIBLE, backprompt

def get_backprompt(pddl_domain, pddl_problem, llm_response, llm_plan_so_far):
    
    step_prompt = get_initial_prompt(pddl_domain, pddl_problem, llm_plan_so_far)
    FEASIBLE, backprompt = get_planning_constraints(llm_response, llm_plan_so_far)
    
    if not FEASIBLE:
        step_prompt += backprompt
    
    return step_prompt

def parse_pddl_action(llm_response):
    llm_response = llm_response.split('\n')
    llm_response = [line.strip() for line in llm_response]
    llm_response = [line for line in llm_response if line]
    return llm_response

def main():

    args = parser.parse_args()
    
    save_dir = f'./llm_modulo_results/{args.llm_model}/Mario-8x11/pddl/variation_{args.variation}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = open(f'{save_dir}/log.txt', 'w')
    plan_file = f'{save_dir}/llm_plan.txt'
    sys.stdout = log_file
    
    conv = Conversation(args.llm_model, temp=0.0)

    pddl_domain_text = open('./llm_modulo/mario_domain.pddl', 'r').read()
    pddl_problem_text = open('./llm_modulo/mario_problem.pddl', 'r').read()
    
    llm_plan_so_far = []
    
    for i in range(args.num_agent_steps):
        
        FEASIBLE = False
        response = ''
        tried_actions = []
        
        for j in range(args.num_backprompt_steps):
            if len(tried_actions) == 0:
                print('[STEP-PROMPTING]---->\n')
                prompt = get_initial_prompt(pddl_domain_text, pddl_problem_text, llm_plan_so_far)
            else:
                print('[BACK-PROMPTING]---->\n')
                prompt = get_backprompt(pddl_domain_text, pddl_problem_text, response, llm_plan_so_far)
                
            print(prompt)
            
            response = conv.llm_actor(prompt, stop=["\n"]).lower()
            print('LLM Response:', response)
            llm_action = parse_pddl_action(response)
            
            FEASIBLE, _ = get_planning_constraints(llm_action[0], llm_plan_so_far)
            
            if FEASIBLE:
                llm_plan_so_far.append(llm_action[0])
                break
            else:
                tried_actions.append(llm_action)
                
        if not FEASIBLE:
            print('LLM could not find a feasible action.')
            break
        
        if llm_plan_so_far[-1] == '(unlock_door)':
            print('LLM has found the solution plan.')
            break
            
    with open(plan_file, 'w') as f:
        for line in llm_plan_so_far:
            f.write("%s\n" % line)
            
    print('LLM plan has been saved to:', plan_file)
    log_file.close()
    
    
if __name__ == "__main__":
    main()
