import warnings
import numpy as np
import pandas
import torch
import math

from entities.components import Components

class ReplayBuffer:
    def __init__(self, max_size = -1):
        self.state = pandas.DataFrame(columns=["observations", "actions", "selected_actions", "rewards", "next_observations"])
        self.max_size = max_size
        self.components = Components.value_list()
        self.dist = torch.zeros((len(self.components), len(self.components)))
        self.dist_obs = []
    
    def update_dist(self, observation):
        self.dist_obs.append(observation)
        nonzero = torch.nonzero(observation).flatten()
        for index_tuple in torch.cartesian_prod(nonzero, nonzero):
            self.dist[index_tuple[0], index_tuple[1]] += 1

    def get_dist_obs(self):
        return self.dist_obs    
    
    def get_dist_probs(self):
        """Returns the conditional probability that we observe broken components together and the probability that a component breaks."""
        clamped_diag = torch.clamp_min(self.dist.diag(), 1)
        return self.dist.div(clamped_diag).t(), self.dist.diag().div(clamped_diag)

    def is_empty(self):
        return len(self.state) == 0
    
    def __len__(self):
        return len(self.state)
    
    def get_state(self):
        return self._to_lists(self.state)

    def set_state(self, observations, actions, selected_actions, rewards, next_observations):
        self.state = pandas.DataFrame(columns=["observations", "actions", "selected_actions", "rewards", "next_observations"])
        self.state["observations"] = observations
        self.state["actions"] = actions
        self.state["selected_actions"] = selected_actions
        self.state["rewards"] = rewards
        self.state["next_observations"] = next_observations

    def _to_lists(self, state: pandas.DataFrame):
        return [
            state["observations"].tolist(),
            state["actions"].tolist(),
            state["selected_actions"].tolist(),
            state["rewards"].tolist(),
            state["next_observations"].tolist(),
        ]

    def add(self, observation, action, selected_action_index, reward, next_observation):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.state = self.state.append({
                "observations": observation,
                "actions": action,
                "selected_actions": selected_action_index,
                "rewards": reward,
                "next_observations": next_observation
            }, ignore_index=True).reset_index(drop=True)
        if self.max_size > 0 and self.max_size < len(self):
            self.state = self.state.iloc[1:]

    def balanced_batch(self, batch_size, random=True):
        positive = self.state[self.state["rewards"] > 0]
        negative = self.state[self.state["rewards"] <= 0]
        if random:
            min_size = min(len(positive), len(negative))
            indices_p = np.random.choice(
                    len(positive),
                    math.ceil(min(batch_size / 2, min_size)),
                    replace=False,
                    #p=(np.arange(len(self.observations)) + 1)/len(self.observations)
                )

            indices_n = np.random.choice(
                    len(negative),
                    math.floor(min(batch_size / 2, min_size)),
                    replace=False,
                    #p=(np.arange(len(self.observations)) + 1)/len(self.observations)
                )
            indices = positive.iloc[indices_p].index.tolist() + negative.iloc[indices_n].index.tolist()
            if len(indices) == 0:
                return None, None, None, None, None
            np.random.default_rng().shuffle(indices)
            indexed = self.state.loc[indices]
            return list(map(torch.stack, self._to_lists(indexed)))
        raise NotImplementedError()

    def positive_batch(self, batch_size, random=True):
        positive = self.state[self.state["rewards"] > 0]
        if len(positive) == 0:
            return None, None, None, None, None
        if random:
            indices_p = np.random.choice(
                    len(positive),
                    min(batch_size, len(positive)),
                    replace=False,
                    #p=(np.arange(len(self.observations)) + 1)/len(self.observations)
            )
            np.random.shuffle(indices_p)
            indexed = positive.iloc[indices_p]
            return list(map(torch.stack, self._to_lists(indexed)))
        raise NotImplementedError()

    def get_batch(self, batch_size, random=True, positive=False, balanced=False):
        if len(self) == 0:
            return None, None, None, None, None
        if positive:
            return self.positive_batch(batch_size, random=random)
        if balanced:
            return self.balanced_batch(batch_size, random=random)
        if random:
            indices = np.random.choice(
                len(self),
                min(batch_size, len(self)),
                replace=False,
                #p=(np.arange(len(self.observations)) + 1)/len(self.observations)
            )
            lists = self._to_lists(self.state.iloc[indices])
        else:
            last_index = min(len(self) - batch_size,0)
            lists = self._to_lists(self.state.iloc[last_index:])
        return list(map(torch.stack, lists))
