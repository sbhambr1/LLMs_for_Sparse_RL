import gymnasium as gym
from minigrid.wrappers import StochasticActionWrapper, SymbolicObsWrapper


def make_env(env_key, seed=None, render_mode=None, stochastic=False, symbolic=False):
    env = gym.make(env_key, render_mode=render_mode)
    if symbolic:
        env = SymbolicObsWrapper(env)
    if stochastic:
        env = StochasticActionWrapper(env, prob=0.9)
    env.reset(seed=seed)
    return env
