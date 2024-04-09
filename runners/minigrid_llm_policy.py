import os
import pickle as pkl
import gymnasium as gym
from utils.conversation import Conversation

class MinigridLLMPolicy:
    
    def __init__(self, llm_model, env) -> None:
        self.llm_model = llm_model
        self.env = env
        self.conversation = Conversation(llm_model=self.llm_model)
        
    def _construct_prompt(self, observation, backprompt):
        prompt = f"Agent is in room {observation['agent_pos']} facing {observation['agent_dir']}. The next action that the agent should take is:"
        return NotImplementedError
    
    def _preprocess_response(self, response):
        return NotImplementedError
    
    def construct_backprompt(self, observation, action):
        return NotImplementedError
    
    def get_action(self, observation, backprompt):
        prompt = self._construct_prompt(observation, backprompt)
        response = self.conversation.llm_actor(prompt, stop=["\n"], role="user")
        response = self._preprocess_response(response)
        return response
    
    def save(self, filename):
        self.conversation.save(filename)
        
    
if __name__ == "__main__":
    llm_model = "gpt-3.5-turbo"
    env_name = "MiniGrid-DoorKey-5x5-v0"
    env = gym.make(env_name)
    policy = MinigridLLMPolicy(llm_model, env)
    
    episodes = 0
    done = False
    episode_history = []
    while episodes < 10:
        episode_trajectory = []
        observation = env.reset()
        while not done:
            action = policy.get_action(observation, backprompt)
            observation, reward, done, info = env.step(action)
            backprompt = policy.construct_backprompt(observation, action)
            episode_trajectory.append((observation, action, reward, done, info, backprompt))
            if done:
                llm_policy = episode_trajectory
                break
        episode_history.append(episode_trajectory)
        episodes += 1
        
    data_dir = "./storage/minigrid_llm_policy"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    save_dir = data_dir + f"/{env_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    policy.save(save_dir + f"/{llm_model}.pkl")
    print("Policy saved successfully!")
    
    episode_history_save_file = save_dir + f"/{llm_model}_episode_history.pkl"
    with open(episode_history_save_file, 'wb') as f:
        pkl.dump(episode_history, f)
    print("Episode history saved successfully!")
    
    llm_policy_save_file = save_dir + f"/{llm_model}_llm_policy.pkl"
    with open(llm_policy_save_file, 'wb') as f:
        pkl.dump(llm_policy, f)
    print("LLM policy saved successfully!")

    print("Done!")