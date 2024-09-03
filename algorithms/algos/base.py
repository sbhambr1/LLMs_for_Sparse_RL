from abc import ABC, abstractmethod
import numpy as np
import torch

from algorithms.format import default_preprocess_obss
from algorithms.utils.dictlist import DictList
from algorithms.utils.penv import ParallelEnv


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : LLM trajectory used for reward shaping function
            a trajectory that shapes the reward for all the same transitions
            that are stored in the buffer.
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        
    # def potential_based_reward_reshaping(self, obs_t, action_t, reward_t):
    #     """
    #     Reshape the reward based on the LLM trajectory.

    #     Parameters:
    #     ----------
    #     obss : 128 observations, len(obss) = 128
    #     actions : 128 actions, len(actions) = 128
    #     rewards : 128 rewards, len(rewards) = 128

    #     Returns:
    #     -------
    #     reshaped_reward : 128 reshaped rewards, len(reshaped_reward) = 128
    #     """
    
    #     traj_len = len(obs_t)
    #     shaped_rewards_t = []
    #     for i in range(traj_len):
    #         if i == 0:
    #             prev_state_action_potential = 0
    #         else:
    #             prev_state = obs_t[i-1]
    #             prev_action = action_t[i-1]
    #             prev_state_action_potential = 0
    #             for j in range(len(self.reshape_reward)):
    #                 if prev_state['image'] == self.reshape_reward[j][0]['image'] and prev_action == self.reshape_reward[j][1]:
    #                     prev_state_action_potential = self.reshape_reward[j][2]
    #                     break
    #         current_state = obs_t[i]
    #         current_action = action_t[i]
    #         current_state_action_potential = 0
    #         for j in range(len(self.reshape_reward)):
    #             if current_state['image'] == self.reshape_reward[j][0]['image'] and current_action == self.reshape_reward[j][1]:
    #                 current_state_action_potential = self.reshape_reward[j][2]
    #                 break
    #         reward_t = reward_t + current_state_action_potential - (1/self.discount)*prev_state_action_potential
    #         shaped_rewards_t.append(reward_t)
    #     return torch.tensor(shaped_rewards_t, device=self.device, dtype=torch.float)
                    
    def potential_based_reward_reshaping(self, obss, actions, rewards):
        """
        Reshape the reward based on the LLM trajectory.

        Parameters:
        ----------
        obss : 128 observations across 16 environments, shape (128, 16)
        actions : 128 actions across 16 environments, shape (128, 16)
        rewards : 128 rewards across 16 environments, shape (128, 16)

        Returns:
        -------
        reshaped_reward : 128 reshaped rewards across 16 environments, shape (128, 16)
        """
        num_envs = len(obss[0])
        shaped_rewards = np.zeros((len(obss), num_envs))
        
        def shape_trajectory_rewards(obs_t, action_t, reward_t):
            traj_len = len(obs_t)
            shaped_rewards_t = []
            for i in range(traj_len):
                if i == 0:
                    prev_state_action_potential = 0
                else:
                    prev_state = obs_t[i-1]
                    prev_action = action_t[i-1]
                    prev_state_action_potential = 0
                    for j in range(len(self.reshape_reward)):
                        prev_state_match = prev_state['image'] == self.reshape_reward[j][0]['image']
                        prev_action_match = prev_action == self.reshape_reward[j][1]
                        if prev_state_match.all() and prev_action_match.item():
                            prev_state_action_potential = self.reshape_reward[j][2]
                            break
                current_state = obs_t[i]
                current_action = action_t[i]
                current_state_action_potential = 0
                for j in range(len(self.reshape_reward)):
                    state_match = current_state['image'] == self.reshape_reward[j][0]['image']
                    action_match = current_action == self.reshape_reward[j][1]
                    if state_match.all() and action_match.item():
                        current_state_action_potential = self.reshape_reward[j][2]
                        break
                shaped_reward = reward_t[i].item() + current_state_action_potential - (1/self.discount)*prev_state_action_potential
                shaped_rewards_t.append(shaped_reward)
            return shaped_rewards_t
        
        def get_trajectory_data(env_idx, obss, actions, rewards):
            obs_t = []
            action_t = []
            reward_t = []
            for i in range(len(obss)):
                obs_t.append(obss[i][env_idx])
                action_t.append(actions[i][env_idx])
                reward_t.append(rewards[i][env_idx])
            return obs_t, action_t, reward_t
                
        for i in range(num_envs):
            # shape rewards for each agent-environment interaction by taking the trajectory of the agent in the i-th environment
            obss_t, actions_t, rewards_t = get_trajectory_data(i, obss, actions, rewards)
            shaped_rewards[:, i] = shape_trajectory_rewards(obss_t, actions_t, rewards_t)
                
        return torch.tensor(shaped_rewards, device=self.device, dtype=torch.float)

    def reward_reshaping(self, obs, action, reward, done):
        """
        Reshape the reward based on the LLM trajectory.

        Parameters:
        ----------
        obs : dict
            the observation
        action : int
            the action
        reward : float
            the reward
        done : bool
            whether the episode is done

        Returns:
        -------
        reshaped_reward : float
            the reshaped reward for parallel environments -> self.rewards[i]
        """
        # for i in range(len(self.reshape_reward)):
        #     # if not done: # do not shape reward for last transition # expt (14)15
        #     comparison_state = self.reshape_reward[i][0]['image'] == obs['image']
        #     final_state_potential = self.reshape_reward[-1][2]
            
        #     if comparison_state.all() and self.reshape_reward[i][1] == action:
        #         current_state_potential = self.reshape_reward[i][2]
        #         reward = reward + final_state_potential - current_state_potential
        #         return reward
        
        # return reward 
        
        for i in range(len(self.reshape_reward)):
            # if not done: # do not shape reward for last transition # expt (14)15
            comparison_state1 = self.reshape_reward[i][0]['image'] == obs['image']
            if comparison_state1.all() and self.reshape_reward[i][1] == action:
                # print('[RESHAPING]')
                # return reward+0.1 # base case expt 9 
                # return reward + self.reshape_reward[i][2] # expt 10
                # alpha = 0.5 # expt 11=0.5, expt 12=0.1, expt 13=0.9, expt 14=0
                
                #implementing PBRS
                alpha = 0.5
                return alpha*reward + (1-alpha)*self.reshape_reward[i][2]
        return reward
    
    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()

            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            
            self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            
        # Perform PBRS (s,a) reward shaping here since we need to traverse within k environments to get consecutive state, action pairs

        self.rewards = self.potential_based_reward_reshaping(self.obss, self.actions, self.rewards)

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
