import json
# from utils.conversation import Conversation
# from utils.ratelimits import *

class LLM_Summarizer():
    
    def __init__(self, model, agent_replay_buffer) -> None:
        self.model = model
        self.agent_replay_buffer = agent_replay_buffer
        self.summary_history = {
            "episode": [], # Episode number
            "episode_length": [], # Episode length
            "summary": [], # Summarized text
            "goal_reached": [], # Goal reached or not
        }
        self.json_path = "summary_history.json" #TODO: Use domain name arguments to specify the path

    def _save_summary(self):
        """
        Save summary history to a json file
        """
        with open(self.json_path, "w") as f:
            json.dump(self.summary_history, f)
        
    def _get_prompt(self, episode):
        """
        Input: 
            - episode: Episode number
            - agent_replay_buffer: Agent replay buffer
        Output:
            - prompt: Prompt for the episode
        """
        return NotImplementedError
    
    def _preprocess(self, text):
        """
        Input: 
            - text: Text to be preprocessed
        Output:
            - preprocessed_text: Preprocessed text
        """
        return NotImplementedError        
        
    def summarize(self, episode_start_index, episode_end_index):
        """
        Input: 
            - model: Language model
            - agent_replay_buffer: Agent replay buffer
        Output:
            - summary: Summarized text of the last episode from agent replay buffer
        """
        return None
    
    def get_relabel_indices(self, episode_summary, episode_start_index, episode_end_index):
        """
        Input: 
            - agent_replay_buffer: Agent replay buffer
        Output:
            - relabel_indices: Indices of episodes to be relabeled
        """
        return None