import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size = -1):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.max_size = max_size

    def add(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

        if self.max_size > 0 and self.max_size < len(self.observations):
            self.observations.pop(0)
            self.rewards.pop(0)
            self.actions.pop(0)

    def get_batch(self, batch_size):
        indices = np.random.choice(
            len(self.observations),
            min(batch_size, len(self.observations)),
            replace=False
        )
        actions = []
        observations = []
        rewards = []
        for index in indices:
            actions.append(self.actions[index])
            observations.append(self.observations[index])
            rewards.append(self.observations[index])
        actions = torch.stack(actions)
        selected_actions = torch.stack(selected_actions)
        observations = torch.stack(observations)
        rewards = torch.stack(rewards)
        return observations, actions, rewards
