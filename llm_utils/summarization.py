import os
import json
from llm_utils.conversation import Conversation
# from utils.ratelimits import *

class LLM_Summarizer():
    
    def __init__(self, model, agent_replay_buffer, map_size, experiment_name) -> None:
        self.model = model
        self.agent_replay_buffer = agent_replay_buffer
        self.summary = []
        self.experiment_name = experiment_name
        self.map_size = map_size
        self.json_path = f"./runners/llm_summaries/lake_{self.map_size}x{self.map_size}{self.experiment_name}/"
        if not os.path.exists(self.json_path):
            os.makedirs(self.json_path)

        self.json_save_file = f"{self.json_path}llm_summary_history.json"
        self.llm_conversation = Conversation(self.model)

    def _store_and_dump(self, episode, episode_length, summary, goal_reached, preprocessed_response, end_episode):
        # Create a dictionary with the provided data
        data = {
            "episode": episode,
            "episode_length": episode_length,
            "summary": summary,
            "goal_reached": goal_reached,
            "preprocessed_response": preprocessed_response
        }

        # Append the data to the summary list
        self.summary.append(data)

        # Write the summary list to the JSON file
        with open(self.json_save_file, 'a') as f:
            # If the file is empty, write an opening bracket
            if f.tell() == 0:
                f.write('[\n')
            
            
            if end_episode:
                f.write(json.dumps(data) + '\n')
                f.write(']')
            else:
                f.write(json.dumps(data) + ',\n')
            
        
    def _get_prompt(self, trajectory, success, examples):
        """
        Input: 
            - trajectory: Trajectory of the episode
            - success: Whether the goal was reached or not
            - examples: examples of bad trajectories duplicating actions, not using after construction.
        Output:
            - prompt: Prompt for the episode after adding an example of good trajectory and a bad trajectory
        """
        
        domain_description = "[DOMAIN] You are in a grid world which involves moving around a house from Entry to the Patio without hitting any dead ends in the house. The game starts with the you at the Entry location of this house grid world with the Patio located somewhere and is currently unknown.\n"
        action_description = "[ACTION] You can take one of the following actions: 'left', 'down', 'right', 'up'. You make moves until you reach the goal, hit a dead end, or find the Patio.\n"
        task_description = "[TASK] You will be provided with a trajectory consisting of states and actions taken at those states, and if the goal was reached or not. Your task is to identify the bad actions taken in the trajectory. A bad action is one that does not make sense based on the task of finding a way to the Patio. Format your answer as shown in the following examples, and do not provide anything else in your response.\n"
        example_description = "[EXAMPLES] Here are some examples of trajectories:\n"
        example1_description = "[Trajectory]: \nI am at the Entry of the house, and took the action to go right.\nI am moving in the house, and took the action to go left.\nI am at the Entry of the house, and took the action to go down.\nI am moving in the house, and took the action to go down.\nI am in the Bathroom of the house, and took the action to go up.\nI am moving in the house, and took the action to go up.\nI am at the Entry of the house, and took the action to go down.\nI am moving in the house, and took the action to go down.\nI am in the Bathroom of the house, and took the action to go down.\nI hit a dead end in the house.\n[Goal reached]: False.\n[Bad actions]: ['taking a up again at Bathroom', 'taking a down again at Entry', ‘taking a down at Bathroom’].\n"
        # example2_description = "[Trajectory]: \nI am at the Entry of the house, and took the action to go right.\nI am moving in the house, and took the action to go right.\nI am in the Bedroom of the house, and took the action to go left.\nI am moving in the house, and took the action to go down.\nI hit a dead end in the house.\n[Goal reached]: False.\n[Bad actions]: ['taking a left again at Bedroom'].\n"
        example2_description = "[Trajectory]: \nI am at the Entry of the house, and took the action to go right.\nI am moving in the house, and took the action to go right.\nI am in the Bedroom of the house, and took the action to go down.\nI am moving in the house, and took the action to go down.\n I am in the Kitchen of the house, and took the action to go right. I hit a dead end in the house.\n[Goal reached]: False.\n[Bad actions]: ['taking a right at Kitchen'].\n"
        trajectory1 = "\n".join(trajectory)
        current_trajectory = "[Current Trajectory]: \n" + trajectory1 + "\n[Goal reached]: " + str(success) + "\n"
        
        final_prompt = domain_description + '\n\n' + action_description + '\n\n' + task_description + '\n\n' + example_description + example1_description + '\n\n' + example2_description + '\n\n' + current_trajectory
        
        return final_prompt
    
    def _get_response(self, prompt):
        """
        Input: 
            - prompt: Prompt for the episode
        Output:
            - response [list]: Response from the LLM includes the list of good actions (preprocessed).
        """
        response = self.llm_conversation.llm_actor(prompt, stop=".")
        return response
    
    def _preprocess(self, text):
        """
        Input: 
            - text: Text to be preprocessed
        Output:
            - preprocessed_text: Preprocessed text
        """
        return NotImplementedError        
    
    def _find_index_of_one(self, tensor):
        # Find the index of the first occurrence of '1'
        index = (tensor == 1).nonzero(as_tuple=False)
        
        # Check if '1' exists in the tensor
        if len(index) == 0:
            return None  # If '1' is not present, return None
        
        return index.item()
    
    def _remove_consecutive_duplicates(self, lst):
        if not lst:
            return []
        
        result = [lst[0]]  # Start with the first element
        indices_selected = [0]
        
        for i in range(1, len(lst)):
            # Append the element if it's different from the preceding one
            if lst[i] != lst[i - 1]:
                result.append(lst[i])
                indices_selected.append(i)
    
        return result, indices_selected
    
    def _preprocess_response(self, response):
        """
        Input: 
            - response: Response from the LLM
        Output:
            - preprocessed_response: Preprocessed response
        Example:
            - response: "['taking a up again at Bathroom', 'taking a down again at Entry']." 
            - preprocessed_response: ['taking a up again at Bathroom', 'taking a down again at Entry']
        """
        response = response.lower()
        bad_actions_index = response.find("[bad actions]:")
        bad_actions_part = response[bad_actions_index + len('[bad actions]:'):].strip()
        bad_actions_part = bad_actions_part.strip("[]").strip()
        preprocessed_response = [s.strip() for s in bad_actions_part.split(",")]
        return preprocessed_response
    
    def construct_trajectory(self, positions, actions):
        """
        Input:
            - positions: List of positions
            - actions: List of actions
        Output:
            - trajectory: List of text templates
        Notes:
            - Current grid layout for home environement (4x4 grid):
            Entry   ,     -      , Bedroom,   -
                -   ,     x      ,   -    ,   x
            Bathroom,     -      , Kitchen,   x
                x   , Living Room,   -    , Patio 
        """
        
        def calculate_row_col(index, num_cols=4):
            row = index // num_cols
            col = index % num_cols
            return row, col
        
        action_dictionary = {
            0: "left",
            1: "down",
            2: "right",
            3: "up"
        }
        
        trajectory = []
        for index in range(len(positions)):
            state = positions[index]
            row, col = calculate_row_col(state)
            action = actions[index]
            action_text = action_dictionary[action]
            
            if row == 0:
                if col == 0:
                    trajectory.append(f"I am at the Entry of the house, and took the action to go {action_text}.")
                elif col == 2:
                    trajectory.append(f"I am in the Bedroom of the house, and took the action to go {action_text}.")
                else:
                    trajectory.append(f"I am moving in the house, and took the action to go {action_text}.")
            elif row == 1:
                if col == 1 or col == 3:
                    trajectory.append(f"I hit a dead end in the house.")
                else:
                    trajectory.append(f"I am moving in the house, and took the action to go {action_text}.")
            elif row == 2:
                if col == 0:
                    trajectory.append(f"I am in the Bathroom of the house, and took the action to go {action_text}.")
                elif col == 2:
                    trajectory.append(f"I am in the Kitchen of the house, and took the action to go {action_text}.")
                elif col == 3:
                    trajectory.append(f"I hit a dead end in the house.")
                else:
                    trajectory.append(f"I am moving in the house, and took the action to go {action_text}.")
            elif row == 3:
                if col == 0:
                    trajectory.append(f"I hit a dead end in the house.")
                elif col == 1:
                    trajectory.append(f"I am in the Living Room of the house, and took the action to go {action_text}.")
                elif col == 3:
                    trajectory.append(f"I have reached the Patio of the house.")
                else:
                    trajectory.append(f"I am moving in the house, and took the action to go {action_text}.")
                
        return trajectory
            
    def construct_example(self, episode_start_index, episode_end_index):
        """
        Input:
            - episode_start_index: Start index of the episode in the agent replay buffer
            - episode_end_index: End index of the episode in the agent replay buffer
        Output:
            - example: Example of the episode in the form of text
        """
        
        positions = []
        actions = []
        for index in range(episode_end_index-episode_start_index+1):
            state = self.agent_replay_buffer.states[episode_start_index+index]
            positions.append(self._find_index_of_one(state))
            actions.append(self.agent_replay_buffer.actions[episode_start_index+index])

        # remove duplicates from positions and actions
        positions_unique, indices = self._remove_consecutive_duplicates(positions)
        positions_unique.append(positions[-1]) if positions[-1] != positions_unique[-1] else None
        actions_unique = []
        for idx in indices:
            if idx == 0:
                continue
            else:
                actions_unique.append(actions[idx-1])
        actions_unique.append(actions[-1])

        # construct trajectory
        trajectory = self.construct_trajectory(positions_unique, actions_unique)
        success = self.agent_replay_buffer.rewards[-1] == 1
        if not success:
            trajectory.append(f"I hit a dead end in the house.")
        
        example_trajectory = "\n".join(trajectory)
        example = f"[Trajectory]: \n{example_trajectory}\n[Goal reached]: {success}"
        
        return example
    
    def summarize(self, episode_start_index, episode_end_index, examples, episode_num, end_episode=False):
        """
        Input: 
            - episode_start_index: Start index of the episode in the agent replay buffer
            - episode_end_index: End index of the episode in the agent replay buffer
        Output:
            - summary: Summarized text of the last episode from agent replay buffer
        """
        
        # build trajectory including state and action taken at that state - in text
        # current templates: state- 'I was at {location} in the grid, and took {action}.'
        
        positions = []
        actions = []
        for index in range(episode_end_index-episode_start_index+1):
            state = self.agent_replay_buffer.states[episode_start_index+index]
            positions.append(self._find_index_of_one(state))
            actions.append(self.agent_replay_buffer.actions[episode_start_index+index])

        # remove duplicates from positions and actions
        positions_unique, indices = self._remove_consecutive_duplicates(positions)
        positions_unique.append(positions[-1]) if positions[-1] != positions_unique[-1] else None
        actions_unique = []
        for idx in indices:
            if idx == 0:
                continue
            else:
                actions_unique.append(actions[idx-1])
        actions_unique.append(actions[-1])

        # construct trajectory
        trajectory = self.construct_trajectory(positions_unique, actions_unique)
        success = self.agent_replay_buffer.rewards[-1] == 1
        
        # construct prompt
        prompt = self._get_prompt(trajectory, success, examples)
        
        # get LLM response
        episode_summary = self._get_response(prompt)
        
        # send the preprocessed response (indices of good actions) to the agent replay buffer
        preprocessed_response = self._preprocess_response(episode_summary)
        
        self._store_and_dump(episode_num, episode_end_index - episode_start_index + 1, episode_summary, success, preprocessed_response, end_episode)
        
        return positions, actions, preprocessed_response            
        
    def get_relabel_indices(self, episode_start_index, episode_end_index, positions, actions, episode_summary):
        """
        Input: 
            - trajectory: Trajectory of the current episode in text (List of strings)
            - episode_summary: Summary of the episode in text (List of strings)
        Output:
            - relabel_indices: Indices of episodes to be relabeled
        """
        relabel_indices = []
        
        state, action = [], []
        for string in episode_summary:
            
            if 'entry' in string:
                state.append(0)
            elif 'bedroom' in string:
                state.append(2)
            elif 'bathroom' in string:
                state.append(8)
            elif 'kitchen' in string:
                state.append(10)
            elif 'living room' in string:
                state.append(13)
        
            if 'left' in string:
                action.append(0)
            elif 'down' in string:
                action.append(1)
            elif 'right' in string:
                action.append(2)
            elif 'up' in string:
                    action.append(3)
                
        pairs = []
        for pair in zip(state, action):
            pairs.append(pair)
            
        for pair in pairs:
            for idx, item in enumerate(zip(positions, actions)):
                if pair[0] == item[0] and pair[1] == item[1]:
                    relabel_indices.append(idx)
                    
        for i in range(len(relabel_indices)):
            relabel_indices[i] += episode_start_index
            assert relabel_indices[i] >= episode_start_index and relabel_indices[i] <= episode_end_index
                
        return relabel_indices