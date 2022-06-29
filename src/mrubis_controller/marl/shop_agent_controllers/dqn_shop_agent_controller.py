from curses import raw
from typing import Dict, List, Union
import torch
from entities.components import Components
from entities.observation import Action, RawAction, SystemObservation
from entities.reward import Reward
from entities.shop import Shop

from marl.mrubis_data_helper import has_shop_remaining_issues
from marl.DQN import LinearNeuralNetwork


class DQNShopAgentController():

    def __init__(
            self,
            shops: List[str],
            model: LinearNeuralNetwork
    ):
        self.shops = shops
        self.alpha = 0.95
        # Replay buffer training hyper parameters
        self.delta = 0.95
        self.discount = 0.9  # 0.01
        self.alpha_rb = 0.99
        self.model = {shop: model().clone() for shop in shops}
        self.old_copy = {}
        self.optimizer = {shop: torch.optim.Adam(
            self.model[shop].parameters(), lr=0.0001) for shop in shops}
        self.copy_current_model()

    def reset_optimizers(self):
        pass

    def copy_current_model(self):
        for shop in self.shops:
            self.old_copy[shop] = self.model[shop].clone()

    def choose_actions(self, observations: SystemObservation) -> Dict[Shop, RawAction]:
        actions = {}
        self.reset_optimizers()
        index = 0
        for shop_name, shop_observation in observations.shops.items():
            if not has_shop_remaining_issues(observations, shop_name):
                continue
            encoded_observation = shop_observation.encode_to_tensor()
            qvalue_tensor = self.model[shop_name](encoded_observation)

            # action_tensor = self.actors[shop_name](
            #    encoded_observation)  # VORSICHT
            highest_qvalue = qvalue_tensor.max().item()
            action, component_index = Components.max_from_tensor(qvalue_tensor)
            action_with_highest_qvalue = qvalue_tensor.argmax().item()
            #action, component_index = Components.from_tensor(action_tensor)
            actions[shop_name] = RawAction(
                action=Action(shop_name, action, highest_qvalue),
                action_tensor=qvalue_tensor,
                expected_utility_tensor=action_with_highest_qvalue,
                action_index=component_index,
                observation_tensor=encoded_observation
            )
            index += 1

        return actions

    def learn(self, action: dict[Shop, RawAction], reward: Reward, observations: SystemObservation, next_observation: SystemObservation):
        for shop_name, raw_action in action.items():
            reward_tensor = torch.tensor(reward[shop_name])
            next_observation_tensor = next_observation.shops[shop_name].encode_to_tensor(
            )

            observations_tensor = observations.shops[shop_name].encode_to_tensor(
            )
            next_utility_tensor = self.model[shop_name](  # self.old_copy
                next_observation_tensor).max()
            loss = torch.pow((reward_tensor + self.discount * next_utility_tensor) -
                             self.model[shop_name](observations_tensor).max(), 2)

            loss.backward()
            self.optimizer[shop_name].step()

    def learn_from_replaybuffer(self, batch_size: int = 1):
        pass
