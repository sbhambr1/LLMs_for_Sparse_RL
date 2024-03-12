import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

def test_taxi_environment():

    # Create the taxi-v3 environment
    env = gym.make('Taxi-v3')

    # Create the DQN agent
    model = DQN("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Test the agent
    # obs = env.reset()
    # for _ in range(100):
    #     action, _ = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         break

    # Evaluate the agent

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_taxi_environment()