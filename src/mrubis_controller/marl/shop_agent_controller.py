from typing import Dict, Iterable, Union
import numpy as np
import torch
import wandb
from entities.components import Components
from entities.observation import Action, RawAction, SystemObservation
from entities.reward import Reward
from entities.shop import Shop

from marl.agent.actor import LSTMActor, LinearActor
from marl.agent.critic import EmbeddingCritic, LinearConcatCritic, WeightedEmbeddingCritic
from marl.mrubis_data_helper import has_shop_remaining_issues
from marl.replay_buffer import ReplayBuffer

class ShopAgentController:
    advantage_loss = True
    def __init__(
        self,
        actor: Union[LSTMActor, LinearActor],
        critic: Union[EmbeddingCritic, WeightedEmbeddingCritic, LinearConcatCritic],
        shops: Iterable[str],
    ):
        self.shops = list(shops)
        self.actors = {shop: actor.clone() for shop in shops}
        self.critics = {shop: critic.clone() for shop in shops}
        self.actor_optimizers = {shop: torch.optim.SGD(self.actors[shop].parameters(), lr=1e-4) for shop in shops}
        self.critic_optimizers = {shop: torch.optim.SGD(self.critics[shop].parameters(), lr=1e-4) for shop in shops}
        self.alpha = 0.95
        self.replay_buffers = {shop: ReplayBuffer() for shop in shops}

        # Replay buffer training hyper parameters
        self.gamma = 0.99
        self.alpha_rb = 0.2

        self.visited_shop = np.zeros(len(shops), dtype=np.bool_)

    def reset_optimizers(self):
        for optimizer in self.actor_optimizers.values():
            optimizer.zero_grad()
        for optimizer in self.actor_optimizers.values():
            optimizer.zero_grad()

    def choose_actions(self, observations: SystemObservation) -> Dict[Shop, RawAction]:
        actions = {}
        self.reset_optimizers()
        for shop_name, shop_observation in observations.shops.items():
            if not has_shop_remaining_issues(observations, shop_name):
                continue
            encoded_observation = shop_observation.encode_to_tensor()
            action_tensor = self.actors[shop_name](encoded_observation)
            expected_utility = self.critics[shop_name](encoded_observation, action_tensor.detach())
            action, component_index = Components.from_tensor(action_tensor)
            actions[shop_name] = RawAction(
                action=Action(shop_name, action, expected_utility.item()),
                action_tensor=action_tensor,
                expected_utility_tensor=expected_utility,
                action_index=component_index,
                observation_tensor=encoded_observation
            )
        return actions

    def learn(self, action: dict[Shop, RawAction], reward: Reward, next_observation: SystemObservation):
        for shop_name, raw_action in action.items():
            reward_tensor = torch.tensor(reward[shop_name])
            next_observation_tensor = next_observation.shops[shop_name].encode_to_tensor()
            next_action_tensor = self.actors[shop_name](next_observation_tensor)
            next_utility_tensor = self.critics[shop_name](next_observation_tensor, next_action_tensor.detach())

            # critic_loss = torch.pow(reward_tensor + self.alpha * (next_utility_tensor.detach() - raw_action.expected_utility_tensor), 2)
            critic_loss = torch.pow(reward_tensor - raw_action.expected_utility_tensor, 2)
            critic_loss.backward()
            self.critic_optimizers[shop_name].step()
            # Advantage. Q_t+1 + R - Q_t
            advantage = reward_tensor + 0.99 * next_utility_tensor - raw_action.expected_utility_tensor
            # advantage = reward_tensor - raw_action.expected_utility_tensor
            advantage = advantage.detach()
            if self.advantage_loss:
                # Actor_loss = Q_t - alpha * log(pi(a_t|s_t))
                actor_loss = advantage * raw_action.action_tensor[raw_action.action_index].log()
            else:
                actor_loss = raw_action.expected_utility_tensor.detach() - self.alpha * raw_action.action_tensor[raw_action.action_index].log()
            actor_loss.backward()
            self.actor_optimizers[shop_name].step()
            self.replay_buffers[shop_name].add(raw_action.observation_tensor, raw_action.action_tensor, torch.tensor(raw_action.action_index), reward_tensor, next_observation_tensor)

    def learn_from_replaybuffer(self, batch_size: int = 1):
        for shop in self.shops:
            if not self.visited_shop[self.shops.index(shop)]:
                continue
            observation, action, selected_action, reward, next_observation = self.replay_buffers[shop].get_batch(batch_size)
            critic = self.critics[shop]
            actor = self.actors[shop]
            critic_optimizer = self.critic_optimizers[shop]
            actor_optimizer = self.actor_optimizers[shop]
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            next_actions_tensor = actor(next_observation).detach()
            selected_action_indices = Components.index_of_tensor(next_actions_tensor)
            selected_actions_probability = torch.take_along_dim(next_actions_tensor, selected_action_indices, dim=1)
            # DDPG Y
            # y = reward + self.gamma * critic(next_observation, next_actions_tensor)
            # SAC Y
            y = reward + self.gamma * (critic(next_observation, next_actions_tensor) - self.alpha_rb * torch.log(selected_actions_probability))
            # Update Q
            critic_loss = torch.pow((critic(observation, action) - y), 2).sum().divide(batch_size)
            critic_loss.backward()
            critic_optimizer.step()
            # Update π
            new_chosen_action_probabilities: torch.Tensor = actor(observation) # batch of probability distributions
            chosen_action_indices = Components.index_of_tensor(new_chosen_action_probabilities) # batch of indices of actions
            probability_of_selected_actions = torch.take(new_chosen_action_probabilities, chosen_action_indices) # batch of probability of chosen action
            actor_loss: torch.Tensor = (critic(observation, action) - self.alpha_rb * torch.log(probability_of_selected_actions)).sum().divide(batch_size)
            actor_loss.backward()
            actor_optimizer.step()
            wandb.log({
                f"{shop} critic_loss": critic_loss.item(),
                f"{shop} actor_loss": actor_loss.item()
            })
    
    def learn_from_replaybuffer(self, batch_size: int = 1):
        for shop in self.shops:
            if not self.visited_shop[self.shops.index(shop)]:
                continue
            observation, action, selected_action, reward, next_observation = self.replay_buffers[shop].get_batch(batch_size)
            critic = self.critics[shop]
            actor = self.actors[shop]
            critic_optimizer = self.critic_optimizers[shop]
            actor_optimizer = self.actor_optimizers[shop]
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            next_actions_tensor = actor(next_observation).detach()
            selected_action_indices = Components.index_of_tensor(next_actions_tensor)
            selected_actions_probability = torch.take_along_dim(next_actions_tensor, selected_action_indices, dim=1)
            # DDPG Y
            y = reward + self.gamma * critic(next_observation, next_actions_tensor)
            # SAC Y
            # y = reward + self.gamma * (critic(next_observation, next_actions_tensor) - self.alpha_rb * torch.log(selected_actions_probability))
            # Update Q
            critic_loss = torch.pow((critic(observation, action) - y), 2).sum().divide(batch_size)
            critic_loss.backward()
            critic_optimizer.step()
            # Update π
            new_chosen_action_probabilities: torch.Tensor = actor(observation) # batch of probability distributions
            chosen_action_indices = Components.index_of_tensor(new_chosen_action_probabilities) # batch of indices of actions
            probability_of_selected_actions = torch.take(new_chosen_action_probabilities, chosen_action_indices) # batch of probability of chosen action
            actor_loss: torch.Tensor = (critic(observation, action) - self.alpha_rb * torch.log(probability_of_selected_actions)).sum().divide(batch_size)
            actor_loss.backward()
            actor_optimizer.step()
            wandb.log({
                f"{shop} critic_loss": critic_loss.item(),
                f"{shop} actor_loss": actor_loss.item()
            })

    def add_to_replaybuffer(self, action: dict[Shop, RawAction], reward: Reward, next_observation: SystemObservation):
        for shop_name, raw_action in action.items():
            self.visited_shop[self.shops.index(shop_name)] = True
            reward_tensor = torch.tensor(reward[shop_name])
            next_observation_tensor = next_observation.shops[shop_name].encode_to_tensor()
            self.replay_buffers[shop_name].add(
                raw_action.observation_tensor.detach(),
                raw_action.action_tensor.detach(),
                torch.tensor(raw_action.action_index).detach(),
                reward_tensor.detach(),
                next_observation_tensor.detach()
            )
