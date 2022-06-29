from curses import raw
from typing import Dict, List, Union
import torch
from entities.components import Components
from entities.observation import Action, RawAction, SystemObservation
from entities.reward import Reward
from entities.shop import Shop

from marl.agent.actor import LSTMActor, LinearActor
from marl.agent.critic import EmbeddingCritic, LinearConcatCritic, WeightedEmbeddingCritic
from marl.mrubis_data_helper import has_shop_remaining_issues
from marl.replay_buffer import ReplayBuffer
from abc import ABC, abstractmethod


class AbstractShopAgentController(ABC):
    advantage_loss = True

    def __init__(
        self,
        shops: List[str],
    ):
        self.shops = shops
        self.alpha = 0.95
        # Replay buffer training hyper parameters
        self.delta = 0.95
        self.discount = 0.01
        self.alpha_rb = 0.99

    @abstractmethod
    def reset_optimizers(self):
        pass

    @abstractmethod
    def choose_actions(self, observations: SystemObservation) -> Dict[Shop, RawAction]:
        pass

    @abstractmethod
    def learn(self, action: dict[Shop, RawAction], reward: Reward, next_observation: SystemObservation):
        pass

    @abstractmethod
    def learn_from_replaybuffer(self, batch_size: int = 1):
        pass
