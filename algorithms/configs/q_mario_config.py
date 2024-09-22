import numpy as np

from algorithms.configs import Agent_Config, State_Represent


class Q_Learning_Config(Agent_Config):
    def __init__(self):
        super(Q_Learning_Config, self).__init__()
        self.max_step = 500000      # max number of total training steps
        self.n_episode = np.inf     # max number of total training episodes
        self.max_episode_len = 700  # max steps in an episode
        self.gamma = 0.99
        self.lr = 0.1
        self.lr_decay = True
        self.max_epsilon = 1.0      # exploration factor
        self.min_epsilon = 0.05     # exploration factor
        self.epsilon_decay = 0.95       # exploration factor decay rate
        self.decay_when_success = True      # only decrease exploration factor upon success
        self.EPSILON = 1e-6         # for numerical stability
        self.replay_buffer_size = int(5e5)
        self.buffer_batch_size = 64
        self.use_sil_buffer = True      # whether to use a separate buffer to store past successful trajectories
        self.sil_batch_size = 64        # batch size when sampling from the SIL buffer
        self.sil_buffer_size = int(1e5)
        self.update_start_from = self.buffer_batch_size + 1
        self.train_freq = 1
        self.render_freq = 100      # for visualization
        self.preprocess_experience_func = None
        # z config
        self.n_episode_per_skill = np.inf
        self.k = 3      # total number of skills to learn per subgoal
        
class Q_Baseline_Config(Q_Learning_Config):
    """
    The config of the Q-Learning baseline
    """
    def __init__(self):
        super(Q_Baseline_Config, self).__init__()

        self.n_episode = 100000
        self.max_step = 500000
        self.greedy_episode = np.inf    # no greedy episode
        self.max_episode_len = 2000
        self.lr = 0.1
        self.preprocess_experience_func = None
        self.render = True
        self.render_success = True
        self.test = False