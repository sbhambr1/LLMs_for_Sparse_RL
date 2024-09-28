import os
import sys
import argparse
sys.path.insert(0,os.getcwd())
from utils.conversation import Conversation

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--llm_model", type=str, default="gpt-3.5-turbo", help="Model to use for LLM")
parser.add_argument("--variation", type=int, default=1, help="Variation of the LLM query if seed != 0.")

def get_initial_prompt(pddl_domain, pddl_problem):
    
    TASK_DESC = "Here is a pddl domain, a planning problem. Provide the plan for the query problem. Provide only the pddl syntax for the plan where each action is represented as (ACTION_NAME OBJECTS).\n\n"
    
    DOMAIN_DESC = f"[DOMAIN]\n{pddl_domain}\n\n"
    PROBLEM_DESC = f"[PROBLEM]\n{pddl_problem}\n\n"
    QUERY_DESC = "[YOUR RESPONSE]\n"
    
    step_prompt = TASK_DESC + DOMAIN_DESC + PROBLEM_DESC + QUERY_DESC
    return step_prompt

def parse_pddl_plan(llm_response):
    llm_response = llm_response.split('\n')
    llm_response = [line.strip() for line in llm_response]
    llm_response = [line for line in llm_response if line]
    return llm_response

def main():

    args = parser.parse_args()
    
    save_dir = f'./vanilla_llm_results/{args.llm_model}/Household/pddl/variation_{args.variation}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = open(f'{save_dir}/log.txt', 'w')
    plan_file = f'{save_dir}/llm_plan.txt'
    sys.stdout = log_file
    
    conv = Conversation(args.llm_model)

    pddl_domain_text = open('./llm_modulo/household_domain.pddl', 'r').read()
    pddl_problem_text = open('./llm_modulo/household_problem.pddl', 'r').read()
    
    initial_prompt = get_initial_prompt(pddl_domain_text, pddl_problem_text)

    response = conv.llm_actor(initial_prompt, stop=["\n"]).lower()

    print(initial_prompt)
    print('LLM Response:', response)
    print('-----------------')
    
    llm_plan = parse_pddl_plan(response)
    print('LLM Plan:', llm_plan)
    print('-----------------')
    
    with open(plan_file, 'w') as f:
        for line in llm_plan:
            f.write("%s\n" % line)
            
    log_file.close()
    
if __name__ == "__main__":
    main()