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

class LLM_Modulo_Minecraft_PDDL:
    
    def __init__(self) -> None:
        self.num_woods_collected = 0
        self.num_woods_processed = 0
        self.stick_made = False
        self.plank_made = False

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
        
        valid_actions = ['(get_wood)', '(get_processed_wood)', '(make_stick)', '(make_plank)', '(make_ladder)']
        
        if llm_response not in valid_actions:
            FEASIBLE = False
            backprompt = 'Your response: ' + llm_response + '\n\n' + 'The action provided is not a valid action. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
        else:
            if '(get_wood)' in llm_response:
                self.num_woods_collected += 1
                FEASIBLE = True
                    
            elif '(get_processed_wood)' in llm_response:
                if self.num_woods_collected > 0:
                    FEASIBLE = True
                    self.num_woods_collected -= 1
                    self.num_woods_processed += 1
                else:
                    FEASIBLE = False
                    backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                    
            elif '(make_stick)' in llm_response:
                if self.num_woods_processed > 0:
                    FEASIBLE = True
                    self.num_woods_processed -= 1
                    self.stick_made = True
                else:
                    FEASIBLE = False
                    backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                    
            elif '(make_plank)' in llm_response:
                if self.num_woods_processed > 0:
                    FEASIBLE = True
                    self.num_woods_processed -= 1
                    self.plank_made = True
                else:
                    FEASIBLE = False
                    backprompt = 'Your plan so far: ' + str(llm_plan_so_far) + '\n\n' + 'Your response: ' + llm_response + '\n\n' + 'The action provided is not feasible. Please choose a valid action from the list ' + str(valid_actions) + '\n.'
                    
            elif '(make_ladder)' in llm_response:
                if self.stick_made and self.plank_made:
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
    
    save_dir = f'./llm_modulo_results/{args.llm_model}/Minecraft/pddl/variation_{args.variation}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = open(f'{save_dir}/log.txt', 'w')
    plan_file = f'{save_dir}/llm_plan.txt'
    sys.stdout = log_file
    
    conv = Conversation(args.llm_model, temp=0)

    pddl_domain_text = open('./llm_modulo/minecraft_domain_relaxed.pddl', 'r').read()
    pddl_problem_text = open('./llm_modulo/minecraft_problem_relaxed.pddl', 'r').read()
    
    llm_plan_so_far = []
    
    runner = LLM_Modulo_Minecraft_PDDL()
    
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
        
        if llm_plan_so_far[-1] == '(make_ladder)':
            print('LLM has found the solution plan.')
            break
            
    with open(plan_file, 'w') as f:
        for line in llm_plan_so_far:
            f.write("%s\n" % line)
            
    print('LLM plan has been saved to:', plan_file)
    log_file.close()
    
    
if __name__ == "__main__":
    main()