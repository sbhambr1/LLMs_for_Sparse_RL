import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
import warnings
warnings.filterwarnings("ignore")

class TestLakeDQN:
    def __init__(self) -> None:
        pass
    
    def create_dqn_agent(self, env):
        # Create the DQN agent with replay buffer size of 1000
        model = DQN("MlpPolicy", env, buffer_size=1000, verbose=1)
        return model

    def train_dqn_agent(self, model, total_episodes, episode_length):
        # Train the agent for the specified number of episodes
        model.learn(total_timesteps=total_episodes * episode_length)
        
    def run_dqn_agent(self, env, total_episodes, episode_length, model):
        for episode in range(total_episodes):
            obs = env.reset()
            episode_history = {
                "obs": [],
                "action": [],
                "reward": []
            }
            for step in range(episode_length):
                if isinstance(obs, tuple):
                    obs = obs[0]
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(int(action))
                episode_history["obs"].append(obs)
                episode_history["action"].append(action)
                episode_history["reward"].append(reward)
                if done:
                    print("Episode terminated")
                    break

    def test_dqn_agent(self, model, env):
        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    def update_replay_buffer(self, model, env, total_episodes, episode_length):
        # Update the replay buffer with updated reward values
        for episode in range(total_episodes):
            obs = env.reset()
            for step in range(episode_length):
                if isinstance(obs, tuple):
                    obs = obs[0]
                action, _ = model.predict(obs)
                obs, reward, done, truncation, info = env.step(int(action))
                # Update the reward value in the replay buffer
                model.replay_buffer.add(obs, action, reward, obs, done)

    def main(self):
        # Create the taxi-v3 environment
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
        # env = DummyVecEnv([lambda: env])
        # env = VecCheckNan(env, raise_exception=True)
        # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # Create the DQN agent
        model = self.create_dqn_agent(env)

        # Update the replay buffer
        total_episodes = 1000
        episode_length = 100
        # self.update_replay_buffer(model, env, total_episodes, episode_length)

        # Train the agent
        self.train_dqn_agent(model, total_episodes, episode_length)

        # Run the agent
        # self.run_dqn_agent(env, total_episodes, episode_length, model)
        
        # Test the agent
        self.test_dqn_agent(model, env)

        env.close()

if __name__ == "__main__":
    test = TestLakeDQN()
    test.main()