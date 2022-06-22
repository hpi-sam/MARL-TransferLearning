import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size = -1):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.max_size = max_size

    def add(self, observation, action, reward, next_obersvation):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obersvation)

        if self.max_size > 0 and self.max_size > len(self.observations):
            self.observations.pop(0)
            self.rewards.pop(0)
            self.actions.pop(0)
            self.next_observations.pop(0)

    def get_batch(self, batch_size):
        indices = np.random.choice(
            len(self.observations),
            min(batch_size, len(self.observations)),
            replace=False
        )
        actions = []
        observations = []
        rewards = []
        next_observations = []
        for index in indices:
            actions.append(self.actions[index])
            observations.append(self.observations[index])
            rewards.append(self.observations[index])
            next_observations.append(self.next_observations[index])
        actions = torch.stack(actions, dim=1)
        observations = torch.stack(observations, dim=1)
        rewards = torch.stack(rewards, dim=1)
        next_observations = torch.stack(next_observations, dim=1)
        return observations, actions, rewards, next_observations
