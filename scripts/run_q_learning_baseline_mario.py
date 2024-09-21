from utils.env_mario import Env_Mario
from example.configs.q_baseline_config import Q_Baseline_Config
from algorithms.algos.q_learning import Q_Learning
from algorithms.utils.experiment_manager import Wandb_Logger


def main():
    env = Env_Mario(success_reward=1)
    config = Q_Baseline_Config()
    logger = Wandb_Logger(proj_name='ASGRL', run_name='household-q-baseline') if config.args.use_wandb else None
    agent = Q_Learning(env, config, logger=logger)
    agent.train()


if __name__ == '__main__':
    main()