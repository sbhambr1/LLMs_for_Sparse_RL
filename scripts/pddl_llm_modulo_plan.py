import os
import sys
import random
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

class LLM_Modulo_Household_PDDL:
    
    def __init__(self) -> None:
        self.carrying_target_key_0 = False
        self.door_0_unlocked = False
        self.carrying_target_key_1 = False
        self.door_1_unlocked = False
        self.is_charged = False
        
    def get_initial_prompt(self, pddl_domain, pddl_problem, llm_plan_so_far):
        
        TASK_DESC = "Here is a pddl domain, a planning problem. Provide only the next action for the query problem. Provide only the pddl syntax for the plan where the action is represented as (ACTION_NAME OBJECTS). Do not provide anything else in your response.\n\n"
        
        DOMAIN_DESC = f"[DOMAIN]\n{pddl_domain}\n\n"
        PROBLEM_DESC = f"[PROBLEM]\n{pddl_problem}\n\n"    
        QUERY_DESC = "[YOUR RESPONSE]\n"
        
        if llm_plan_so_far == []:
            step_prompt = TASK_DESC + DOMAIN_DESC + PROBLEM_DESC + QUERY_DESC
        else:    
            step_prompt = TASK_DESC + DOMAIN_DESC + PROBLEM_DESC + 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + QUERY_DESC
        return step_prompt

    def get_planning_constraints(self, llm_response, llm_plan_so_far):
        
        FEASIBLE = False
        backprompt = ''
        
        valid_actions = ['(get_key0)', '(get_key1)', '(open_door0)', '(open_door1)', '(is_charged)', '(goal)']
        
        random.shuffle(valid_actions)
        
        if llm_response not in valid_actions:
            FEASIBLE = False
            backprompt = 'Your response: ' + llm_response + '\n\n' + 'The action provided is not a valid action. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
        else:
            if '(get_key0)' in llm_response:
                self.carrying_target_key_0 = True
                FEASIBLE = True
                
            elif '(open_door0)' in llm_response:
                if self.carrying_target_key_0:
                    self.door_0_unlocked = True
                    FEASIBLE = True
                else:
                    FEASIBLE = False
                    backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                    
            elif '(get_key1)' in llm_response:
                if self.door_0_unlocked:
                    self.carrying_target_key_1 = True
                    FEASIBLE = True
                else:
                    FEASIBLE = False
                    backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                    
            elif '(open_door1)' in llm_response:
                if self.carrying_target_key_1:
                    self.door_1_unlocked = True
                    FEASIBLE = True
                else:
                    FEASIBLE = False
                    backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                    
            elif '(is_charged)' in llm_response:
                if self.door_1_unlocked and self.door_0_unlocked:
                    self.is_charged = True
                    FEASIBLE = True
                else:
                    FEASIBLE = False
                    backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                    
            elif '(goal)' in llm_response:
                if self.is_charged:
                    FEASIBLE = True
                else:
                    FEASIBLE = False
                    backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
        
        return FEASIBLE, backprompt

    def get_backprompt(self, pddl_domain, pddl_problem, llm_response, llm_plan_so_far):
        
        step_prompt = self.get_initial_prompt(pddl_domain, pddl_problem, llm_plan_so_far)
        FEASIBLE, backprompt = self.get_planning_constraints(llm_response, llm_plan_so_far)
        
        if not FEASIBLE:
            step_prompt += backprompt
        
        return step_prompt

    def parse_pddl_action(self, llm_response):
        if '(' in llm_response and ')' in llm_response:
            llm_response = llm_response.split('(')[1].split(')')
        if len(llm_response) > 1:
            llm_response = llm_response[0]
        if ' ' in llm_response:
            llm_response = llm_response.split(' ')[0]
        if llm_response[0] != '(':
            llm_response = '(' + llm_response
        if llm_response[-1] != ')':
            llm_response = llm_response + ')'
        return llm_response

def main():

    args = parser.parse_args()
    
    save_dir = f'./llm_modulo_results/{args.llm_model}/Household/pddl/variation_{args.variation}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = open(f'{save_dir}/log.txt', 'w')
    plan_file = f'{save_dir}/llm_plan.txt'
    sys.stdout = log_file
    
    conv = Conversation(args.llm_model)

    pddl_domain_text = open('./llm_modulo/household_domain.pddl', 'r').read()
    pddl_problem_text = open('./llm_modulo/household_problem.pddl', 'r').read()
    
    llm_plan_so_far = []
    
    runner = LLM_Modulo_Household_PDDL()
    
    for i in range(args.num_agent_steps):
        
        FEASIBLE = False
        response = ''
        tried_actions = []
        
        for j in range(args.num_backprompt_steps):
            if len(tried_actions) == 0:
                print('[STEP-PROMPTING]---->\n')
                prompt = runner.get_initial_prompt(pddl_domain_text, pddl_problem_text, llm_plan_so_far)
            else:
                print('[BACK-PROMPTING]---->\n')
                prompt = runner.get_backprompt(pddl_domain_text, pddl_problem_text, response, llm_plan_so_far)
                
            print(prompt)
            
            response = conv.llm_actor(prompt, stop=["\n"]).lower()
            print('LLM Response:', response)
            llm_action = runner.parse_pddl_action(response)
            
            FEASIBLE, _ = runner.get_planning_constraints(llm_action, llm_plan_so_far)
            
            if FEASIBLE:
                llm_plan_so_far.append(response)
                break
            else:
                tried_actions.append(response)
                
        if not FEASIBLE:
            print('LLM could not find a feasible action.')
            break
        
        if '(goal)' in llm_plan_so_far[-1]:
            print('LLM has found the solution plan.')
            break
            
    with open(plan_file, 'w') as f:
        for line in llm_plan_so_far:
            f.write("%s\n" % line)
            
    print('LLM plan has been saved to:', plan_file)
    log_file.close()
    
    
if __name__ == "__main__":
    main()