from typing import Dict, Iterable, Union
import torch
import wandb
from entities.components import Components
from entities.observation import Action, RawAction, SystemObservation
from entities.reward import Reward
from entities.shop import Shop
import numpy as np
from marl.agent.actor import LSTMActor, LinearActor
from marl.agent.critic import EmbeddingCritic, LinearConcatCritic, WeightedEmbeddingCritic
from marl.mrubis_data_helper import has_shop_remaining_issues
from marl.replay_buffer import ReplayBuffer

import torch.nn.functional as F


class Agent():
    def __init__(self,
                 shops: Iterable[str],
                 actor: Union[LSTMActor, LinearActor],
                 critic: Union[EmbeddingCritic,
                               WeightedEmbeddingCritic, LinearConcatCritic]
                 ):

        self.shops = shops
        self.actor = actor
        self.num_components = 18
        self.critic = critic
        self.actor_optimizer = torch.optim.SGD(
            self.critic.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.SGD(
            self.critic.parameters(), lr=1e-4)
        self.alpha = 0.95
        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99
        self.alpha_rb = 0.2
        self.visited_shop = np.zeros(len(shops), dtype=np.bool_)
        self.sampled_actions: Dict[Shop, 'list[int]'] = {}

    def reset_optimizers(self):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

    def reset_sampled_actions(self):
        self.sampled_actions = {}

    def choose_actions(self, observations: SystemObservation) -> Dict[Shop, RawAction]:
        actions = {}
        self.reset_optimizers()
        for shop_name, shop_observation in observations.shops.items():
            if not has_shop_remaining_issues(observations, shop_name):
                continue
            encoded_observation = shop_observation.encode_to_tensor()
            action_tensor = self.actor(encoded_observation)
            expected_utility = self.critic(
                encoded_observation, action_tensor.detach())
            action, component_index = Components.from_tensor(action_tensor)

            # Only sample actions that have not been tryed out so far yet, since this never makes sense
            """
            if shop_name in self.sampled_actions.keys():
                while component_index in self.sampled_actions[shop_name]:
                    action, component_index = Components.from_tensor(
                        action_tensor)
                self.sampled_actions[shop_name].append(component_index)
            else:
                self.sampled_actions[shop_name] = [component_index]
            """
            actions[shop_name] = RawAction(
                action=Action(shop_name, action, expected_utility.item()),
                action_tensor=action_tensor,
                expected_utility_tensor=expected_utility,
                action_index=component_index,
                observation_tensor=encoded_observation
            )
        return actions

    def learn(self, action: Dict[Shop, RawAction], reward: Reward, next_observation: SystemObservation):
        for shop_name, raw_action in action.items():
            reward_tensor = torch.tensor(reward[shop_name])
            next_observation_tensor = next_observation.shops[shop_name].encode_to_tensor(
            )
            next_action_tensor = self.actor(
                next_observation_tensor)
            next_utility_tensor = self.critic(
                next_observation_tensor, next_action_tensor.detach())

            # critic_loss = torch.pow(reward_tensor + self.alpha * (next_utility_tensor.detach() - raw_action.expected_utility_tensor), 2)
            critic_loss = torch.pow(
                reward_tensor - raw_action.expected_utility_tensor, 2)
            critic_loss.backward()
            self.critic_optimizer.step()
            # Advantage. Q_t+1 + R - Q_t
            advantage = reward_tensor + 0.99 * next_utility_tensor - \
                raw_action.expected_utility_tensor
            # advantage = reward_tensor - raw_action.expected_utility_tensor
            advantage = advantage.detach()
            if self.advantage_loss:
                # Actor_loss = Q_t - alpha * log(pi(a_t|s_t))
                actor_loss = advantage * \
                    raw_action.action_tensor[raw_action.action_index].log()
            else:
                actor_loss = raw_action.expected_utility_tensor.detach(
                ) - self.alpha * raw_action.action_tensor[raw_action.action_index].log()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.replay_buffer.add(raw_action.observation_tensor, raw_action.action_tensor, torch.tensor(
                raw_action.action_index), reward_tensor, next_observation_tensor)

    def learn_from_replaybuffer(self, batch_size: int = 1):
        for shop in self.shops:
            if not self.visited_shop[self.shops.index(shop)]:
                continue
            observation, action, selected_action, reward, next_observation = self.replay_buffer.get_batch(
                batch_size)
            # One hot encode previously chosen action:
            one_hot_actions: torch.Tensor = F.one_hot(
                selected_action, num_classes=self.num_components)
            critic = self.critic
            actor = self.actor
            critic_optimizer = self.critic_optimizer
            actor_optimizer = self.actor_optimizer
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            # Tensor of probability per component e.g. [0.1, 0.5, 0.4]
            next_actions_tensor = actor(next_observation).detach()
            # Index of sampled action to take using probabilties from above e.g. [1]
            selected_action_indices = Components.index_of_tensor(
                next_actions_tensor)
            # One hot encode sampled actions e.g. [0, 1, 0]
            one_hot_sampled_actions: torch.Tensor = F.one_hot(
                selected_action_indices.squeeze(1), num_classes=self.num_components)
            # Probability of chosen action e.g. [0.5]
            selected_actions_probability = torch.take_along_dim(
                next_actions_tensor, selected_action_indices, dim=1)
            # DDPG Y
            # y = reward + self.gamma * critic(next_observation, next_actions_tensor)
            # SAC Y
            y = reward + self.gamma * (critic(next_observation, one_hot_sampled_actions) -
                                       self.alpha_rb * torch.log(selected_actions_probability))
            # Update Q
            critic_loss = torch.pow(
                (critic(observation, one_hot_actions) - y), 2).sum().divide(batch_size)
            critic_loss.backward()
            critic_optimizer.step()
            # Update π
            # Tensor of probability per component e.g. [0.1, 0.5, 0.4]
            new_chosen_action_probabilities: torch.Tensor = actor(
                observation)  # batch of probability distributions
            # Index of sampled action to take using probabilties from above e.g. [1]
            chosen_action_indices = Components.index_of_tensor(
                new_chosen_action_probabilities)  # batch of indices of actions
            # One hot encode sampled actions e.g. [0, 1, 0]
            one_hot_sampled_actions: torch.Tensor = F.one_hot(
                chosen_action_indices.squeeze(1), num_classes=self.num_components)
            # Probability of chosen action e.g. [0.5]
            probability_of_selected_actions = torch.take(
                new_chosen_action_probabilities, chosen_action_indices)  # batch of probability of chosen action
            # TODO What we want is to maximize the probability of the action which leads the highest value given by the critic
            # minimize: - sum over actions( probablity of action_i * critic(observation, action_i))
            component_sum = torch.zeros((observation.shape[0], 1))
            eye = torch.eye(self.num_components)
            eye_batch = eye.repeat([observation.shape[0], 1, 1])
            operation = "mult"
            for i in range(self.num_components):
                if operation == "mult":
                    component_sum -= critic(observation, eye_batch[:, :, i]) * torch.log(
                        new_chosen_action_probabilities[:, i]).unsqueeze(1)
                else:
                    component_sum += critic(observation, eye_batch[:, :, i]) - torch.log(
                        new_chosen_action_probabilities[:, i])
            actor_loss = component_sum.sum().divide(batch_size)

            #actor_loss: torch.Tensor = (critic(observation, one_hot_sampled_actions) - self.alpha_rb * torch.log(probability_of_selected_actions)).sum().divide(batch_size)
            actor_loss.backward()
            actor_optimizer.step()
            wandb.log({
                f"{shop} critic_loss": critic_loss.item(),
                f"{shop} actor_loss": actor_loss.item()
            })

    # def learn_from_replaybuffer(self, batch_size: int = 1):
    #     for shop in self.shops:
    #         if not self.visited_shop[self.shops.index(shop)]:
    #             continue
    #         observation, action, selected_action, reward, next_observation = self.replay_buffers[shop].get_batch(batch_size)
    #         critic = self.critics[shop]
    #         actor = self.actors[shop]
    #         critic_optimizer = self.critic_optimizers[shop]
    #         actor_optimizer = self.actor_optimizers[shop]
    #         critic_optimizer.zero_grad()
    #         actor_optimizer.zero_grad()
    #         next_actions_tensor = actor(next_observation).detach()
    #         selected_action_indices = Components.index_of_tensor(next_actions_tensor)
    #         selected_actions_probability = torch.take_along_dim(next_actions_tensor, selected_action_indices, dim=1)
    #         # DDPG Y
    #         y = reward + self.gamma * critic(next_observation, next_actions_tensor)
    #         # SAC Y
    #         # y = reward + self.gamma * (critic(next_observation, next_actions_tensor) - self.alpha_rb * torch.log(selected_actions_probability))
    #         # Update Q
    #         critic_loss = torch.pow((critic(observation, action) - y), 2).sum().divide(batch_size)
    #         critic_loss.backward()
    #         critic_optimizer.step()
    #         # Update π
    #         new_chosen_action_probabilities: torch.Tensor = actor(observation) # batch of probability distributions
    #         chosen_action_indices = Components.index_of_tensor(new_chosen_action_probabilities) # batch of indices of actions
    #         probability_of_selected_actions = torch.take(new_chosen_action_probabilities, chosen_action_indices) # batch of probability of chosen action
    #         actor_loss: torch.Tensor = (critic(observation, action) - self.alpha_rb * torch.log(probability_of_selected_actions)).sum().divide(batch_size)
    #         actor_loss.backward()
    #         actor_optimizer.step()
    #         wandb.log({
    #             f"{shop} critic_loss": critic_loss.item(),
    #             f"{shop} actor_loss": actor_loss.item()
    #         })

    def add_to_replaybuffer(self, action: Dict[Shop, RawAction], reward: Reward, next_observation: SystemObservation):
        for shop_name, raw_action in action.items():
            if shop_name not in self.shops:
                continue
            self.visited_shop[self.shops.index(shop_name)] = True
            reward_tensor = torch.tensor(reward[shop_name])
            next_observation_tensor = next_observation.shops[shop_name].encode_to_tensor(
            )
            self.replay_buffer.add(
                raw_action.observation_tensor.detach(),
                raw_action.action_tensor.detach(),
                torch.tensor(raw_action.action_index).detach(),
                reward_tensor.detach(),
                next_observation_tensor.detach()
            )
