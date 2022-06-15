from typing import Dict, List, Union
import torch
from entities.components import Components
from entities.observation import Action, RawAction, SystemObservation
from entities.reward import Reward
from entities.shop import Shop

from marl.agent.actor import LSTMActor, LinearActor
from marl.agent.critic import EmbeddingCritic, LinearConcatCritic, WeightedEmbeddingCritic

class ShopAgentController:
    def __init__(
        self,
        actor: Union[LSTMActor, LinearActor],
        critic: Union[EmbeddingCritic, WeightedEmbeddingCritic, LinearConcatCritic],
        shops: List[str],
    ):
        self.actors = {shop: actor.clone() for shop in shops}
        self.critics = {shop: critic.clone() for shop in shops}
        self.actor_optimizers = {shop: torch.optim.Adam(self.actors[shop].parameters(), lr=0.01) for shop in shops}
        self.critic_optimizers = {shop: torch.optim.Adam(self.critics[shop].parameters(), lr=0.01) for shop in shops}
        self.critic_loss = torch.nn.L1Loss()

    def reset_optimizers(self):
        for optimizer in self.actor_optimizers.values():
            optimizer.zero_grad()
        for optimizer in self.actor_optimizers.values():
            optimizer.zero_grad()

    def choose_actions(self, observations: SystemObservation) -> Dict[Shop, RawAction]:
        actions = {}
        self.reset_optimizers()
        for shop_name, shop_observation in observations.shops.items():
            encoded_observation = shop_observation.encode_to_tensor()
            action_tensor = self.actors[shop_name](encoded_observation)
            expected_utility = self.critics[shop_name](encoded_observation, action_tensor)
            action = Components.from_tensor(action_tensor)
            actions[shop_name] = RawAction(Action(shop_name, action, expected_utility.item()), action_tensor, expected_utility)
        return actions

    def learn(self, action: dict[Shop, RawAction], reward: Reward):
        for shop_name, raw_action in action.items():
            expected_reward = raw_action.expected_utility_tensor
            actual_reward = torch.tensor(reward[shop_name])
            loss = self.critic_loss.forward(expected_reward, actual_reward)
            loss.backward()
            self.actor_optimizers[shop_name].step()
            self.critic_optimizers[shop_name].step()
