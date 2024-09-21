import os
from utils.env_mario import Env_Mario
from algorithms.configs.q_mario_config import Q_Baseline_Config
from algorithms.algos.q_learning import Q_Learning
from algorithms.utils.experiment_manager import Wandb_Logger

os.environ["WANDB_MODE"] = "online"

def main():
    env = Env_Mario(success_reward=1)
    config = Q_Baseline_Config()
    logger = Wandb_Logger(entity_name='llm_modulo_sparse_rl' ,proj_name='neurips_24', run_name='MARIO_q_baseline')
    agent = Q_Learning(env, config, logger=logger)
    agent.train()


if __name__ == '__main__':
    main()