import os
import utils
import argparse
import pickle
from utils.env_craft import Env_Craft
from algorithms.configs.q_minecraft_config import Q_Baseline_Config
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
parser.add_argument("--reshape_reward", type=bool, default=False,
                    help="pass a path to the reward shaping plan if True")
parser.add_argument("--variation", type=int, default=0,
                    help="variation of the reward shaping plan")
parser.add_argument("--llm_model", type=str, default='gpt-3.5-turbo',
                    help="LLM model name")

def main():
    
    args = parser.parse_args()
    utils.seed(args.seed)
    env = Env_Craft(success_reward=1, stochastic=args.stochastic)
    config = Q_Baseline_Config()
    logger = Wandb_Logger(entity_name='llm_modulo_sparse_rl' ,proj_name='neurips_24', run_name='MINECRAFT_q_learning_baseline'+args.additional_info)
    
    if args.reshape_reward:
        file_path = f"./storage/vanilla_llm_visualization/{args.llm_model}/Minecraft/variation_{args.variation}/vanilla_llm_policy.pkl"
        with open(file_path, 'rb') as f:
            llm_rs_policy = pickle.load(f)
        print(f"[INFO] LLM reward shaping plan loaded from {file_path}.\n")
    else:
        llm_rs_policy = None
    
    agent = Q_Learning(env, config, logger=logger, reshape_reward=llm_rs_policy)
    agent.train()


if __name__ == '__main__':
    main()