import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size = -1):
        self.observations = []
        self.actions = []
        self.selected_actions = []
        self.rewards = []
        self.next_observations = []
        self.max_size = max_size
    
    def is_empty(self):
        return len(self.observations) == 0
    
    def __len__(self):
        return len(self.observations)
    
    def get_state(self):
        return self.observations, self.actions, self.selected_actions, self.rewards, self.next_observations

    def set_state(self, observations, actions, selected_actions, rewards, next_observations):
        self.observations = observations
        self.actions = actions
        self.selected_actions = selected_actions
        self.rewards = rewards
        self.next_observations = next_observations

    def add(self, observation, action, selected_action_index, reward, next_obersvation):
        self.observations.append(observation)
        self.actions.append(action)
        self.selected_actions.append(selected_action_index)
        self.rewards.append(reward)
        self.next_observations.append(next_obersvation)

        if self.max_size > 0 and self.max_size < len(self.observations):
            self.observations.pop(0)
            self.rewards.pop(0)
            self.actions.pop(0)
            self.selected_actions.pop(0)
            self.next_observations.pop(0)

    def get_batch(self, batch_size):
        indices = np.random.choice(
            len(self.observations),
            min(batch_size, len(self.observations)),
            replace=False,
            p=(np.arange(self.observations)+1)/len(self.observations)
        )
        actions = []
        selected_actions = []
        observations = []
        rewards = []
        next_observations = []
        for index in indices:
            actions.append(self.actions[index])
            selected_actions.append(self.selected_actions[index])
            observations.append(self.observations[index])
            rewards.append(self.rewards[index])
            next_observations.append(self.next_observations[index])
        actions = torch.stack(actions)
        selected_actions = torch.stack(selected_actions)
        observations = torch.stack(observations)
        rewards = torch.stack(rewards)
        next_observations = torch.stack(next_observations)
        return observations, actions, selected_actions, rewards, next_observations
