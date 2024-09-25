import os
import utils
import argparse
from utils.env_mario import Env_Mario
from algorithms.configs.q_mario_config import Q_Baseline_Config
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
parser.add_argument("--reshape_reward", type=str, default='',
                    help="pass a path to the reward shaping plan if True")

def main():
    
    args = parser.parse_args()
    utils.seed(args.seed)
    env = Env_Mario(success_reward=1, stochastic=args.stochastic)
    config = Q_Baseline_Config()
    logger = Wandb_Logger(entity_name='llm_modulo_sparse_rl' ,proj_name='neurips_24', run_name='MARIO_q_learning_baseline'+args.additional_info)
    llm_rs_policy = None
    agent = Q_Learning(env, config, logger=logger, reshape_reward=llm_rs_policy)
    agent.train()


if __name__ == '__main__':
    main()