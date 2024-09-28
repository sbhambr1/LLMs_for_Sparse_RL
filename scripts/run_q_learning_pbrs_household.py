import os
import utils
import argparse
import pickle
from utils.env_household import Env_Household
from algorithms.configs.q_household_config import Q_Baseline_Config
from algorithms.algos.q_learning import Q_Learning
from algorithms.utils.experiment_manager import Wandb_Logger

os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--stochastic", action='store_true',
                    help="use stochastic environment")
parser.add_argument("--additional_info", type=str, default='',
                    help="additional info for the run")
parser.add_argument("--reshape_reward", type=bool, default=True,
                    help="pass a path to the reward shaping plan if True")
parser.add_argument("--variation", type=int, default=3,
                    help="variation of the reward shaping plan")
parser.add_argument("--llm_model", type=str, default='gpt-4o',
                    help="LLM model name")
parser.add_argument('--llm_plan', type=str, default='llm_modulo', help='LLM plan to use, other option is "llm_modulo"')

def main():
    
    args = parser.parse_args()
    utils.seed(args.seed)
    env = Env_Household(success_reward=1, stochastic=args.stochastic)
    config = Q_Baseline_Config()
    logger = Wandb_Logger(entity_name='llm_modulo_sparse_rl' ,proj_name='neurips_24', run_name='HOUSEHOLD_q_learning_baseline'+args.additional_info)
    
    if args.reshape_reward:
        
        root_dir = os.getcwd()
    
        if args.llm_plan == 'vanilla':
            search_dir = f"{root_dir}/vanilla_llm_results/{args.llm_model}/Household/pddl/variation_{args.variation}/"
        elif args.llm_plan == 'llm_modulo':
            search_dir = f"{root_dir}/llm_modulo_results/{args.llm_model}/Household/pddl/variation_{args.variation}/"
        
        file_path = search_dir + "reward_shaping_with_llm_plan.pkl"
        with open(file_path, 'rb') as f:
            reward_flags = pickle.load(f)
        print(f"[INFO] LLM reward shaping plan loaded from {file_path}.\n")
    else:
        reward_flags = None
    
    agent = Q_Learning(env, config, logger=logger, reshape_reward=reward_flags)
    agent.train()


if __name__ == '__main__':
    main()